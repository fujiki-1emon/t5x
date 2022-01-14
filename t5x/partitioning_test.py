# Copyright 2021 The T5X Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for t5x.partitioning."""

import collections

from absl.testing import absltest
from flax import optim
from flax.linen import partitioning as nn_partitioning
import jax
import numpy as np
from t5x import adafactor
from t5x import partitioning
from t5x import test_utils as ptu
from t5x import train_state

jax.config.parse_flags_with_absl()

mock = absltest.mock
TpuDevice = ptu.TpuDevice
TPUV3_32 = ptu.make_devices(4, 4, 1, 2, kind='TPU v3')


class PartitioningTest(absltest.TestCase):

  @mock.patch('jax.host_count')
  @mock.patch('jax.local_device_count')
  def test_bounds_from_last_device(self, local_device_count, host_count):
    last_device = mock.Mock(coords=(3, 3, 3), core_on_chip=1)
    tpu_bounds = partitioning.bounds_from_last_device(last_device)
    self.assertEqual(tpu_bounds, (4, 4, 4, 2))

    last_device = mock.Mock(spec=[])
    host_count.return_value = 1
    local_device_count.return_value = 4
    non_tpu_bounds = partitioning.bounds_from_last_device(last_device)
    self.assertEqual(non_tpu_bounds, (1, 4))

  @mock.patch('jax.local_device_count')
  def test_get_coords(self, local_device_count):
    device = mock.Mock(coords=(1, 0, 1), core_on_chip=1)
    coords = partitioning.get_coords(device)
    self.assertEqual(coords, (1, 0, 1, 1))

    device = mock.Mock(spec=['process_index', 'id'])
    device.process_index = 1
    device.id = 9
    local_device_count.return_value = 8
    coords = partitioning.get_coords(device)
    self.assertEqual(coords, (1, 1))

  @mock.patch('jax.local_devices')
  @mock.patch('jax.devices')
  @mock.patch('jax._src.lib.xla_bridge.process_index')
  def test_default_mesh(self, process_index_fn, devices_fn, local_devices_fn):
    devices_fn.return_value = TPUV3_32
    local_devices_fn.return_value = [
        d for d in TPUV3_32 if d.process_index == 0
    ]
    process_index_fn.return_value = 0

    global_mesh = partitioning.default_mesh(4)
    self.assertEqual(global_mesh.axis_names, ('data', 'model'))
    self.assertEqual(global_mesh.shape,
                     collections.OrderedDict((('data', 8), ('model', 4))))
    self.assertEqual(global_mesh.size, 32)

    for process_index in (0, 1, 2, 3):
      process_index_fn.return_value = process_index
      local_mesh = global_mesh.local_mesh
      self.assertEqual(local_mesh.axis_names, ('data', 'model'))
      self.assertEqual(local_mesh.shape,
                       collections.OrderedDict((('data', 2), ('model', 4))))
      self.assertEqual(local_mesh.size, 8)

    process_index_fn.return_value = 0
    local_mesh = global_mesh.local_mesh
    lds = np.array([
        [
            TpuDevice(id=0, process_index=0, coords=(0, 0, 0), core_on_chip=0),
            TpuDevice(id=1, process_index=0, coords=(0, 0, 0), core_on_chip=1),
            TpuDevice(id=2, process_index=0, coords=(1, 0, 0), core_on_chip=0),
            TpuDevice(id=3, process_index=0, coords=(1, 0, 0), core_on_chip=1)
        ],
        [
            TpuDevice(id=8, process_index=0, coords=(0, 1, 0), core_on_chip=0),
            TpuDevice(id=9, process_index=0, coords=(0, 1, 0), core_on_chip=1),
            TpuDevice(id=10, process_index=0, coords=(1, 1, 0), core_on_chip=0),
            TpuDevice(id=11, process_index=0, coords=(1, 1, 0), core_on_chip=1)
        ]
    ],
                   dtype=object)
    np.testing.assert_array_equal(local_mesh.devices, lds)

  @mock.patch('jax.local_devices')
  @mock.patch('jax.devices')
  @mock.patch('jax._src.lib.xla_bridge.process_index')
  def test_local_chunker(self, process_index_fn, devices_fn, local_devices_fn):
    devices_fn.return_value = TPUV3_32
    local_devices_fn.return_value = [
        d for d in TPUV3_32 if d.process_index == 0
    ]
    process_index_fn.return_value = 0
    global_mesh = partitioning.default_mesh(4)
    local_chunker = partitioning.LocalChunker(global_mesh)
    self.assertEqual(local_chunker.num_chunks['data'], 4)
    self.assertEqual(local_chunker.num_chunks['model'], 1)

    # Derive the chunk order along the first 'data' dim for testing.
    host_ordering = []
    for d in global_mesh.devices[:, 0]:
      if d.process_index not in host_ordering:
        host_ordering.append(d.process_index)
    process_index_to_data_pos = {
        process_index: idx for idx, process_index in enumerate(host_ordering)
    }

    for process_indexx in (0, 1, 2, 3):
      process_index_fn.return_value = process_indexx
      global_mesh = partitioning.default_mesh(4)
      local_chunker = partitioning.LocalChunker(global_mesh)
      # get expected chunk for 'data' axis.
      expected_chunk = process_index_to_data_pos[process_indexx]
      self.assertEqual(local_chunker.chunk_ids['data'], expected_chunk)
      self.assertEqual(local_chunker.chunk_ids['model'], 0)
      # Sharded along both axes.
      local_chunk_info = local_chunker.get_local_chunk_info((128, 16),
                                                            ['data', 'model'])
      self.assertEqual(local_chunk_info.replica_id, 0)
      self.assertEqual(local_chunk_info.slice,
                       (slice(32 * expected_chunk, 32 *
                              (expected_chunk + 1)), slice(0, 16)))
      # Replicated across first axis.
      local_chunk_info = local_chunker.get_local_chunk_info((128, 16),
                                                            [None, 'model'])
      self.assertEqual(local_chunk_info.replica_id, expected_chunk)
      self.assertEqual(local_chunk_info.slice, (slice(None), slice(0, 16)))

  def test_model_based_partitioner_get_mesh_axes(self):
    AxisMetadata = nn_partitioning.AxisMetadata  # pylint: disable=invalid-name
    rules = (('batch', 'data'), ('embed', None), ('vocab', 'model'), ('mlp',
                                                                      'model'))
    adafactor_rules = {
        'batch': adafactor.FactorDim.NONE,
        'embed': adafactor.FactorDim.ROW,
        'vocab': adafactor.FactorDim.COLUMN,
        'mlp': adafactor.FactorDim.COLUMN,
    }
    ppm = partitioning.ModelBasedPjitPartitioner(
        num_partitions=1, logical_axis_rules=rules)

    target = {
        'logits_dense': np.ones((16, 16), np.float32),
        'mlp': {
            'wo': {
                'kernel': np.ones((32, 16), np.float32)
            }
        }
    }

    def get_axes_spec(factored, momentum):
      opt_def = adafactor.Adafactor(
          learning_rate=0.1,
          factored=factored,
          min_dim_size_to_factor=8,
          beta1=0.1 if momentum else None,
          logical_factor_rules=adafactor_rules)
      optimizer = opt_def.create(target)
      state = train_state.TrainState.from_flax_optimizer(optimizer)
      state = state.replace(
          axes_variables={
              'params_axes': {
                  'logits_dense_axes': AxisMetadata(names=('vocab', 'embed')),
                  'mlp': {
                      'wo': {
                          'kernel_axes': AxisMetadata(names=('embed', 'mlp'))
                      }
                  }
              }
          })
      axis_names = nn_partitioning.get_axis_names(
          state.axes_variables['params_axes'])
      state._optimizer.optimizer_def.set_param_axes(axis_names)
      return ppm.get_mesh_axes(state).state_dict()

    def get_expected_axes_spec(spec_0, spec_1):
      return train_state.TrainState.from_flax_optimizer(
          optimizer=optim.Optimizer(
              # opt_def,
              adafactor.Adafactor(0.1),  # opt_def not compared.
              state=optim.OptimizerState(
                  step=None,
                  param_states={
                      'logits_dense': spec_0,
                      'mlp': {
                          'wo': {
                              'kernel': spec_1
                          }
                      }
                  }),
              target={
                  'logits_dense': partitioning.PartitionSpec('model', None),
                  'mlp': {
                      'wo': {
                          'kernel': partitioning.PartitionSpec(None, 'model')
                      }
                  }
              })).state_dict()

    p0_spec = partitioning.PartitionSpec('model', None)
    p1_spec = partitioning.PartitionSpec(None, 'model')

    # Test quadrant of conditions: factored or not / momentum or not.
    axes_spec = get_axes_spec(factored=True, momentum=False)
    expected_axes_spec = get_expected_axes_spec(
        adafactor._AdafactorParamState(m=None, v=None, v_col=None, v_row=None),
        adafactor._AdafactorParamState(m=None, v=None, v_col=None, v_row=None))
    jax.tree_multimap(self.assertEqual, axes_spec, expected_axes_spec)

    axes_spec = get_axes_spec(factored=True, momentum=True)
    expected_axes_spec = get_expected_axes_spec(
        adafactor._AdafactorParamState(
            m=p0_spec, v=None, v_col=None, v_row=None),
        adafactor._AdafactorParamState(
            m=p1_spec, v=None, v_col=None, v_row=None))
    jax.tree_multimap(self.assertEqual, axes_spec, expected_axes_spec)

    axes_spec = get_axes_spec(factored=False, momentum=True)
    expected_axes_spec = get_expected_axes_spec(
        adafactor._AdafactorParamState(
            m=p0_spec, v=p0_spec, v_col=None, v_row=None),
        adafactor._AdafactorParamState(
            m=p1_spec, v=p1_spec, v_col=None, v_row=None))
    jax.tree_multimap(self.assertEqual, axes_spec, expected_axes_spec)

    axes_spec = get_axes_spec(factored=False, momentum=False)
    expected_axes_spec = get_expected_axes_spec(
        adafactor._AdafactorParamState(
            m=None, v=p0_spec, v_col=None, v_row=None),
        adafactor._AdafactorParamState(
            m=None, v=p1_spec, v_col=None, v_row=None))
    jax.tree_multimap(self.assertEqual, axes_spec, expected_axes_spec)


if __name__ == '__main__':
  absltest.main()
