from qos.types.types import Engine
from typing import Any, Dict, List, Tuple
from qos.types.types import Qernel
import logging
from time import sleep
from qos.types.types import QPU
import numpy as np
from mapomatic import layouts

pipe_name = "multiprog_fifo.pipe"

class Multiprogrammer(Engine):

    timeout = 5
    window_size = 5
    
    def __init__(self) -> None:

        self.queue = []
        self.done_queue = []

    def spatial_utilization(self, q1: Qernel, q2: Qernel, backend: QPU) -> float:
        circ1 = q1.get_circuit()
        circ2 = q2.get_circuit()
        util1 = circ1.num_qubits / backend.num_qubits
        util2 = circ2.num_qubits / backend.num_qubits

        return util1 + util2

    def effective_utilization(self, q1: Qernel, q2: Qernel, backend: QPU) -> float:
        """
        Calculate the effective utilization of a quantum processing unit (QPU) 
        based on the characteristics of two Qernels.

        Args:
            q1 (Qernel): The first Qernel object, representing a quantum program.
            q2 (Qernel): The second Qernel object, representing a quantum program.
            backend (QPU): The quantum processing unit (QPU) backend, providing 
                           hardware specifications such as the number of qubits.

        Returns:
            float: The effective utilization of the QPU, calculated as the sum of 
                   spatial utilization and temporal utilization, expressed as a percentage.

        Notes:
            - Spatial utilization is determined by the Qernel with the maximum
              depth (D_max) relative to the total number of qubits available on
              the backend.
            - Temporal utilization accounts for the shorter-depth Qernel,
              weighted by the ratio of its depth to D_max.
        """
        # Find the Qernel with the maximum depth (D_max)
        circ1 = q1.get_circuit()
        circ2 = q2.get_circuit()
        D1 = circ1.depth()
        D2 = circ2.depth()
        D_max = max((D1, D2))

        # Spatial utilization (from the circuit with maximum depth)
        if D1 >= D2:
            C_max = circ1.num_qubits
            C_other, D_other = circ2.num_qubits, D2
        else:
            C_max = circ2.num_qubits
            C_other, D_other = circ1.num_qubits, D1

        spatial_util = (C_max / backend.num_qubits) * 100

        # Temporal utilization (from shorter circuits, weighted by depth ratio)
        if D_max > 0:
            temporal_util = (D_other / D_max) * (C_other / backend.num_qubits) * 100
        else:
            temporal_util = 0.0

        # Total effective utilization
        u_eff = spatial_util + temporal_util

        return u_eff

    def get_matching_score(self, q1: Qernel, q2: Qernel, backend: QPU, weighted: bool = False, weights: List[float] = []) -> float:
        """
        Calculate the matching score between two Qernels based on various comparison metrics.
        Args:
            q1 (Qernel): The first Qernel to compare.
            q2 (Qernel): The second Qernel to compare.
            backend (QPU): The quantum processing unit (QPU) backend used for evaluation.
            weighted (bool, optional): If True, apply weights to the comparison metrics. Defaults to False.
            weights (List[float], optional): A list of weights for the comparison metrics in the order:
                [effective utilization, entanglement difference, measurement difference, parallelism difference].
                Defaults to an empty list.
        Returns:
            float: The calculated matching score. If weights are provided and valid, the score is a weighted sum
            of the metrics; otherwise, it is the average of the metrics.
        """
        score = 0

        util_eff = self.effective_utilization(q1, q2, backend)
        entanglementDiff = self.entanglementComparison(q1, q2)
        measurementDiff = self.measurementComparison(q1, q2)
        parallelismDiff = self.parallelismComparison(q1, q2)

        if weighted and sum(weights) > 0:            
            score = weights[0] * util_eff + weights[1] * entanglementDiff + weights[2] * measurementDiff + weights[3] * parallelismDiff
        else:
            score = (util_eff + entanglementDiff + measurementDiff + parallelismDiff) / 4
    
        return score
    

    def depthComparison(self, q1: Qernel, q2: Qernel) -> float:
        metadata_1 = q1.get_metadata()
        metadata_2 = q2.get_metadata()

        depthDiff = np.abs(metadata_1["depth"] - metadata_2["depth"])
        
        k = 0.05

        depthFactor = np.exp(-k * depthDiff)
                   
        return depthFactor
    
    def entanglementComparison(self, q1: Qernel, q2: Qernel) -> float:
        metadata_1 = q1.get_metadata()
        metadata_2 = q2.get_metadata()

        entanglement_result = (1 - metadata_1["entanglement_ratio"]) * (1 - metadata_2["entanglement_ratio"])

        return entanglement_result
    
    def measurementComparison(self, q1: Qernel, q2: Qernel) -> float:
        metadata_1 = q1.get_metadata()
        metadata_2 = q2.get_metadata()

        measurement_result = (1 - metadata_1["measurement"]) * (1 - metadata_2["measurement"])

        return measurement_result
    
    def parallelismComparison(self, q1: Qernel, q2: Qernel) -> float:
        metadata_1 = q1.get_metadata()
        metadata_2 = q2.get_metadata()

        parallelism_result = (1 - metadata_1["parallelism"]) * (1 - metadata_2["parallelism"])

        return parallelism_result

    def process_qernels(
        self,
        qernel_dict: Dict[Qernel, List[Tuple[List[int], Any, float]]],
        threshold: float,
        dry_run: bool = False,
        pair_filter: Any = None,
        return_ranked: bool = False,
    ):
        """
        Processes a dictionary of Qernels to compute spatial utilization and matching scores 
        for pairs of Qernels using the same backend. Filters and evaluates pairs based on 
        utilization, matching score, and layout overlap.

        Args:
            qernel_dict (Dict[Qernel, List[Tuple[List[int], Any, float]]]):
                A dictionary where keys are Qernel objects and values are lists of tuples. 
                Each tuple contains:
                    - A list of integers representing the layout.
                    - A backend object (e.g., FakeKolkataV2).
                    - A float representing the estimated fidelity.
            threshold (float): 
                The minimum matching score required to process a pair of Qernels.
            dry_run (bool):
                If True, return the selected pair metadata without invoking policies.
            pair_filter (callable | None):
                Optional predicate to filter candidate pairs. Signature:
                (q1, q2, layout1, layout2, backend) -> bool
            return_ranked (bool):
                If True, return all candidate pairs that pass the threshold,
                ordered by spatial utilization (descending).

        Returns:
            Qernel | tuple | list | None:
                The bundled Qernel, the selected pair metadata when dry_run=True,
                or a ranked list of pair metadata when return_ranked=True.

        Behavior:
            - Computes spatial utilization for pairs of Qernels using the same backend.
            - Keeps pairs with spatial utilization below 0.9.
            - Sorts the filtered pairs by utilization in descending order.
            - Evaluates each pair's matching score and checks if it exceeds the threshold.
            - Applies policies based on layout overlap:
                - Calls `restrict_policy()` if layouts do not overlap.
                - Calls `re_evaluation_policy()` if layouts overlap.
        """
        results = []
        selected_qernel = None
        selected_pair = None
        ranked_pairs = []

        def check_layout_overlap(layout1: List[int], layout2: List[int]) -> bool:
            for q in layout1:
                if q in layout2:
                    return True
            return False

        # Iterate over the dictionary to compute spatial utilization for each pair with the same backend
        for q1, q1_data in qernel_dict.items():
            for q2, q2_data in qernel_dict.items():
                if q1 == q2:
                    continue

                # Check if the backends (str) are the same
                for layout1, backend1, _ in q1_data:
                    for layout2, backend2, _ in q2_data:
                        if backend1 == backend2:
                            if pair_filter and not pair_filter(q1, q2, layout1, layout2, backend1):
                                continue

                            # Compute spatial utilization
                            spatial_util = self.spatial_utilization(q1, q2, backend1)

                            # Filter pairs with utilization > 9
                            if spatial_util < 0.9:
                                results.append((q1, q2, layout1, layout2, spatial_util, backend1))

        # Sort the pairs by utilization in descending order
        results.sort(key=lambda x: x[4], reverse=True)

        # Iterate over the pairs by highest utilization
        for q1, q2, layout1, layout2, spatial_util, backend in results:
            # Compute the matching score for their respective best layout
            matching_score = self.get_matching_score(q1, q2, backend)

            # Check if the matching score is over the threshold
            if matching_score > threshold:
                if return_ranked:
                    ranked_pairs.append(
                        (q1, q2, layout1, layout2, matching_score, spatial_util, backend)
                    )
                    continue
                if dry_run:
                    selected_pair = (q1, q2, layout1, layout2, matching_score, spatial_util, backend)
                    break
                # Check if the layouts have common elements
                if not check_layout_overlap(layout1, layout2):
                    selected_qernel = self.restrict_policy([q1, q2], error_limit=threshold)
                else:
                    selected_qernel = self.re_evaluation_policy(q1, q2)
                if selected_qernel is not None:
                    break

        if return_ranked:
            ranked_pairs.sort(key=lambda x: (x[4], x[5]), reverse=True)
            return ranked_pairs
        if dry_run:
            return selected_pair
        return selected_qernel

    def run(self, qernels: List[Qernel]) -> List[Qernel]:


        return qernels #<--- Comment this out
       
    def results(self) -> None:
        pass


     # Merging policies:

    # 1. Restrist policy: Only merge if the best QPU for two circuits is the same and their best layouts dont overlap
    #   Start by considering one of the new circuits with the oldest circuits on the window
    #   After considering merging the new circuits with each one on the window move the window the number of circuits as the number of circuits that couldnt be merged
    #   Example: The window has 5 circuits E, D, C, B, A and the new circuits are F, G, H.
    #   1. Consider merging F or G or H with E, D, C, B, A, by this order. Lets consider that F could be merged with A
    #   2. The new circuits left are G and H, move the window by two circuits, this is because the new circuits need to enter the window and the size of the window is fixed
    
    def restrict_policy(self, new_qernels: Qernel | List[Qernel] , error_limit: float, matching_cycles=1, max_bundle=2) -> None:
        from qos.multiprogrammer.tools import check_layout_overlap, size_overflow, bundle_qernels

        # This is whole algorithm is very very unclean, but it works for now
        # This compares matching 0 of the incoming qernel with matching 0 of a qernel on the queue, the 1 to 0 then 0 to 1 and so on
        cycles = [(0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (2,1), (1,2), (2,2), (3,0), (0,3), (3,1), (1,3), (3,2), (2,3), (3,3)]



        #There is a caveat here, if a qernel has been merged the matching will always be the same, so it might make some repeated checks

        next = 0

        for q in new_qernels:
            for cycle in range(0, matching_cycles):
                if next == 1:
                    next = 0
                    break
                if self.queue == []:
                    break
                for queue in self.queue[::-1]:
                    incoming_qpu = q.matching[cycles[cycle][0]][1]
                    queue_qpu = queue.matching[cycles[cycle][1]][1] if queue.match == None else queue.match[1]
                    incoming_map = q.matching[cycles[cycle][0]][0]
                    queue_map = queue.matching[cycles[cycle][1]][0] if queue.match == None else queue.match[0]
                    incoming_fid = q.matching[cycles[cycle][0]][2]
                    queue_fid = queue.matching[cycles[cycle][1]][2] if queue.match == None else queue.match[2]
                    #The first element of the matching is the ideal mapping, the second is the qpu and the third is the estimated fidelity
                    if incoming_qpu == queue_qpu and not check_layout_overlap(incoming_map, queue_map) and not size_overflow(q, queue, incoming_qpu) and not (((incoming_fid + queue_fid) / 2) > error_limit):
                                    
                        print("Multiprogramming match found!")
                        tmp = bundle_qernels(q, queue, (incoming_map, incoming_qpu, incoming_fid))
                                    
                        if tmp == 0:
                        # This is the case that the qernel on the queue already was a bundled qernel and the new qernel was just added to the bundle, nothing more to do.
                            tmp = queue
                        
                        else:
                        #This is the case that the qernel on the queue was a normal qernel and a new bundled qernel was created this new qernel is going to substitute the original qernel on the queue
                            self.queue.remove(queue)
                            self.queue.insert(0,tmp)
                                        
                        if len(tmp.src_qernels) >= max_bundle:
                        #If the bundled qernel already has the max bundle size, then it jumps the queue and is scheduled
                            self.done_queue.insert(0,tmp)
                            self.queue.remove(tmp)

                        next = 1
                        break

            self.queue.insert(0,q)

            if len(self.queue) > self.window_size:
                self.done_queue.insert(0,self.queue.pop())

        self.done_queue = self.queue + self.done_queue
        return 0

    def re_evaluation_policy(self, qernel1: Qernel, qernel2, successors=False) -> Qernel:
        from qos.multiprogrammer.tools import bundle_qernels
        from qos.estimator.estimator import Estimator

        bundled_qernel = bundle_qernels(qernel1, qernel2, (0,0))

        estimator = Estimator()
        # Re-run the estimator with the bundled Qernel
        result = estimator.run(bundled_qernel)

        return result

    def _base_policy(self, newQernel: Qernel, error_limit: float) -> None:
        import qos.database as db
        from qos.multiprogrammer.tools import check_layout_overlap

        # self.logger.log(10, "Running Base policy")

        logger = logging.getLogger(__name__)
        logging.basicConfig(level=10)

        window = db.currentWindow()

        for j in newQernel.subqernels:
            for i in window[::-1]:
                for x in i.matching:
                    print(x)
                    if not check_layout_overlap(j.best_layout(), i.best_layout()):
                        logger.log(
                            10, "Possible match found, checking score threshold."
                        )

                        # If the average of the scores is above the threshold fidelidy, then merge
                        if ((j.best_score + i.best_score) / 2) > error_limit:
                            logger.log(10, "Multiprogramming match found!")
                            return 0

        return 0
