import logging

from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.utils import (
    get_inter_core_tiling_size,
)

logger = logging.getLogger(__name__)


class SetFixedAllocationStage(Stage):
    REQUIRED_FIELDS = ("workload", "accelerator")

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        ctx: StageContext,
    ):
        super().__init__(list_of_callables, ctx)
        self.accelerator = self.ctx.require_value("accelerator", self.__class__.__name__)
        self.workload = self.ctx.require_value("workload", self.__class__.__name__)

    def run(self):
        logger.info("Start SetFixedAllocationStage.")
        # Set the performance of all nodes that have a fixed allocation
        self.set_fixed_allocation()
        logger.info("Finished SetFixedAllocationStage.")

        self.ctx.set(workload=self.workload, accelerator=self.accelerator)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def set_fixed_allocation(self):
        for node in self.workload.node_list:
            inter_core_tiling_size = get_inter_core_tiling_size(node)
            if len(node.core_allocation) == inter_core_tiling_size:
                chosen_core_allocation = node.core_allocation[node.group]
                node.set_chosen_core_allocation(chosen_core_allocation)
                # Sanity check: core exists in accelerator
                core = self.accelerator.get_core(chosen_core_allocation)
                assert core is not None, f"Core {chosen_core_allocation} not found in accelerator."
