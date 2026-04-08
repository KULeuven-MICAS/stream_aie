import logging
import os
from typing import TypeAlias

from stream.cost_model.steady_state_scheduler import SteadyStateScheduler
from stream.datatypes import LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.mapping.mapping import Mapping
from stream.opt.allocation.constraint_optimization.config import ConstraintOptStageConfig
from stream.opt.search_space import SearchSpace
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.workload import Workload

logger = logging.getLogger(__name__)

SCHEDULE_ORDER_T: TypeAlias = list[tuple[int, int]]


class ConstraintOptimizationAllocationStage(Stage):
    """
    Class that finds the best workload allocation for the workload using constraint optimization.
    This stage requires a TileAwareLatencyEstimator (from CoreCostEstimationStage) for node latency computation.
    """

    REQUIRED_FIELDS = (
        "workload",
        "accelerator",
        "mapping",
        "fusion_splits",
        "output_path",
    )

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        ctx: StageContext,
    ):
        super().__init__(list_of_callables, ctx)
        self.workload: Workload = self.ctx.get("workload")
        self.accelerator: Accelerator = self.ctx.get("accelerator")
        self.mapping: Mapping = self.ctx.get("mapping")
        self.fusion_splits: dict[LayerDim, int] = self.ctx.get("fusion_splits")
        self.latency_estimator = self.ctx.get("latency_estimator")

        config = self.ctx.get("constraint_opt_config")
        if config is None:
            logger.warning(
                "ConstraintOptimizationAllocationStage: legacy kwargs configuration path is deprecated. "
                "Please pass a ConstraintOptStageConfig. Building config from kwargs for now."
            )
            config = ConstraintOptStageConfig.from_legacy_kwargs(**self.ctx.data)
        self.config = config

        self.output_path = self.ctx.get("output_path")
        self.search_space: SearchSpace | None = self.ctx.get("search_space")

    def run(self):
        logger.info("Start ConstraintOptimizationAllocationStage.")
        workload, scheduler = self.find_best_tensor_transfer_allocation()
        mapping = scheduler.mapping
        self.ctx.set(workload=workload, mapping=mapping, scheduler=scheduler)
        logger.info("End ConstraintOptimizationAllocationStage.")
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def find_best_tensor_transfer_allocation(self):
        """
        Run a simple scheduler that finds the optimal tensor transfer allocation for the workload.
        """
        output_path = os.path.join(self.output_path, "tetra")
        scheduler = SteadyStateScheduler(
            self.workload,
            self.accelerator,
            self.mapping,
            self.fusion_splits,
            nb_cols_to_use=self.config.transfer.nb_cols_to_use,
            output_path=output_path,
            search_space=self.search_space,
            latency_estimator=self.latency_estimator,
        )
        workload = scheduler.run()
        return workload, scheduler

    def is_leaf(self) -> bool:
        return False
