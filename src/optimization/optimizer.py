import itertools
import time
import warnings

import numpy as np
from pulp import LpVariable, LpProblem, LpMaximize, lpSum, LpStatus, PULP_CBC_CMD, GLPK_CMD, GUROBI_CMD
from tqdm import tqdm

from src.data.ihdp_1 import get_outcome
from src.optimization.utils import matrix_lin_space
from src.utils.setup import add_dict


class Optimization_problem():
    """
    Represents an optimization problem for treatment allocation.

    Attributes:
        data: The dataset containing the features and outcomes.
        model: The predictive model used for estimating treatment effects.
        ntreatmentlevels: The number of treatment levels. ($\delta$)
        optimization_settings: Settings for the optimization process.
    """

    def __init__(self, data, model, ntreatmentlevels, optimization_settings):
        """
        Initializes the optimization problem.

        Parameters:
            data: The dataset containing the features and outcomes.
            model: The predictive model used for estimating treatment effects.
            ntreatmentlevels: The number of treatment levels.
            optimization_settings: Settings for the optimization process.
        """
        self.data = data
        self.model = model
        self.ntreatmentlevels = ntreatmentlevels
        self.optimization_settings = optimization_settings

        num_rows = self.data.x_test.shape[0]
        num_cols = self.ntreatmentlevels

        # doses
        self.s = np.zeros((num_rows, num_cols))
        self.s[:, :] = np.linspace(0, 1, num_cols)

        self.ite_est_matrix = np.ones((num_rows, num_cols))
        for col in range(num_cols):
            self.ite_est_matrix[:, col] \
                = self.model.predict(
                self.data.x_test, self.s[:, col],
                np.zeros(num_rows)) - self.model.predict(
                self.data.x_test,
                np.zeros(num_rows) * 0.00001,
                np.zeros(num_rows))  # find more elegant solution to avoid 0

        self.ite_gt_matrix = np.ones((num_rows, num_cols))
        for col in range(num_cols):
            self.ite_gt_matrix[:, col] \
                = get_outcome(
                self.data.x_test, self.s[:, col],
                np.zeros(num_rows)) - get_outcome(
                self.data.x_test,
                np.zeros(num_rows) * 0.00001,
                np.zeros(num_rows))  # find more elegant solution to avoid 0

        # define benefits for uplift
        if optimization_settings["cost_sensitive_optimization"] == False:
            self.b = np.ones(num_rows)  # all entities have the same benefit
            self.b_test = np.ones(num_rows)  # all entities have the same benefit
        elif optimization_settings["cost_sensitive_optimization"] == True:
            self.b = self.data.b  # benefits are dependent on the entity
            self.b_test = self.data.b[:num_rows]  # benefits are dependent on the entity


        # define costs for treatment. This version: cost is linear with dosage (as done in the paper)
        self.c = matrix_lin_space(num_rows, num_cols)

    def solve(self, method="exact", budget=float('inf'), ground_truth=False) -> float:
        """
        Solves the optimization problem using the specified method.

        Parameters:
            method (str): The method to use for solving the problem. Defaults to "exact".
            budget (float): The budget constraint. Defaults to infinity.
            ground_truth (bool): Whether to use ground truth data. Defaults to False.

        Returns:
            float: The objective value of the solution.
        """

        if method == "exact":
            objective, decision_vars = self.solve_exact(budget, ground_truth)

        elif method == "heuristic1":
            objective, decision_vars = self.solve_heuristic1(budget, ground_truth)

        elif method == "heuristic2":
            objective, decision_vars = self.solve_heuristic2(budget, ground_truth)

        else:
            raise NotImplementedError("Only 'exact', 'heuristic1', and 'heuristic2' are implemented.")

        return objective, decision_vars

    def solve_exact(self, budget=float('inf'), ground_truth=False):
        """
        Finds the optimal solution using exact methods.

        Parameters:
            budget (float): The budget constraint. Defaults to infinity.
            ground_truth (bool): Whether to use ground truth data. Defaults to False.

        Returns:
            tuple: The objective value and decision variables.
        """

        # Initialize the model
        problem = LpProblem(name="treatment_allocation", sense=LpMaximize)

        # Define the decision variables
        decision_vars = {(i, j): LpVariable(name=f"d{i}_{j}", lowBound=0, upBound=1, cat='Binary')
                         for i in range(1, self.data.x_test.shape[0] + 1)
                         for j in range(1, self.ntreatmentlevels + 1)}

        # Define the parameters
        # ITE parameters (either estimated or ground truth)
        if ground_truth is False:
            ite_paras = {
                (i, j): self.ite_est_matrix[i - 1, j - 1]
                for i in range(1, self.data.x_test.shape[0] + 1)
                for j in range(1, self.ntreatmentlevels + 1)
            }
            print(f'finding $\stackrel{budget}\Pi^{{exp}}$ with ILP')

        elif ground_truth is True:
            ite_paras = {
                (i, j): self.ite_gt_matrix[i - 1, j - 1]
                for i in range(1, self.data.x_test.shape[0] + 1)
                for j in range(1, self.ntreatmentlevels + 1)
            }
            print(f'finding $\stackrel{budget}\Pi^{{opt}}$ with ILP')

        # dose parameters
        s_paras = {
            (i, j): self.s[i - 1, j - 1]
            for i in range(1, self.data.x_test.shape[0] + 1)
            for j in range(1, self.ntreatmentlevels + 1)
        }

        # benefit parameters
        b_paras = {
            i: self.b[i - 1]
            for i in range(1, self.data.x_test.shape[0] + 1)
        }

        # cost parameters (linear with dosage)
        c_paras = {
            (i, j): self.c[i - 1, j - 1]
            for i in range(1, self.data.x_test.shape[0] + 1)
            for j in range(1, self.ntreatmentlevels + 1)
        }

        # Define the objective function
        problem += lpSum(
            decision_vars[i, j] * ite_paras[i, j] * b_paras[i]
            for i in range(1, self.data.x_test.shape[0] + 1)
            for j in range(1, self.ntreatmentlevels + 1))

        # Define the constraints
        # Budget constraint
        problem += (lpSum(
            decision_vars[i, j] * c_paras[i, j]
            for i in range(1, self.data.x_test.shape[0] + 1)
            for j in range(1, self.ntreatmentlevels + 1)) <= budget)

        # Every i receives exactly one dosage j
        for i in range(1, self.data.x_test.shape[0] + 1):
            problem += lpSum(
                decision_vars[i, j]
                for j in range(1, self.ntreatmentlevels + 1)) == 1

        # Fairness constraints: disparate treatment and disparate outcome
        A = self.optimization_settings["protected_attribute"]  # By default feature '23' in IHDP dataset
        eps_treatment = self.optimization_settings["epsilon_disparate_treatment"]
        eps_outcome = self.optimization_settings["epsilon_disparate_outcome"]

        prot_0_N = np.sum(self.data.x_test[:, A] == 0)
        prot_1_N = np.sum(self.data.x_test[:, A] == 1)

        # fairness constraint: demographic parity approach for disparate treatment
        prot_0_treat_sum = lpSum(
            decision_vars[i, j] * s_paras[i, j]
            for i in range(1, self.data.x_test.shape[0] + 1)
            for j in range(1, self.ntreatmentlevels + 1)
            if self.data.x_test[i - 1, A] == 0
        )

        prot_1_treat_sum = lpSum(
            decision_vars[i, j] * s_paras[i, j]
            for i in range(1, self.data.x_test.shape[0] + 1)
            for j in range(1, self.ntreatmentlevels + 1)
            if self.data.x_test[i - 1, A] == 1)

        problem += (prot_0_treat_sum / prot_0_N) >= (1 - eps_treatment) * (prot_1_treat_sum / prot_1_N)
        problem += (prot_0_treat_sum / prot_0_N) <= (1 + eps_treatment) * (prot_1_treat_sum / prot_1_N)

        # Fairness constraint: demographic parity approach for disparate outcome
        prot_0_ite_sum = lpSum(
            decision_vars[i, j] * ite_paras[i, j]
            for i in range(1, self.data.x_test.shape[0] + 1)
            for j in range(1, self.ntreatmentlevels + 1)
            if self.data.x_test[i - 1, A] == 0)

        prot_1_ite_sum = lpSum(
            decision_vars[i, j] * ite_paras[i, j]
            for i in range(1, self.data.x_test.shape[0] + 1)
            for j in range(1, self.ntreatmentlevels + 1)
            if self.data.x_test[i - 1, A] == 1)

        problem += (prot_0_ite_sum / prot_0_N) >= (1 - eps_outcome) * (prot_1_ite_sum / prot_1_N)
        problem += (prot_0_ite_sum / prot_0_N) <= (1 + eps_outcome) * (prot_1_ite_sum / prot_1_N)

        # Solve the problem
        # status = problem.solve(GUROBI_CMD())
        solver = GUROBI_CMD(msg=1, options=[
            ("OutputFlag", 1),  # Enable output to the console
            ("LogToConsole", 1),  # Ensure log messages are sent to the console
            ("DisplayInterval", 5)  # Set the display interval to 5 seconds
        ])
        status = problem.solve(solver)

        if status == 1:
            print("Optimization problem solved successfully.")

        elif status != 1:
            raise ValueError(f"Optimization problem did not converge. Status: {LpStatus[status]}")


        print(f"Status: {LpStatus[status]}")

        return problem.objective.value(), decision_vars

    def solve_heuristic1(self, budget=float('inf'), ground_truth=False):
        """
        heuristic1: methodology of Olaya et al. (2019) for the optimization problem:
        Optimal treatment for i is the treatment for which \hat{\tau}(i, d) * b(i) is maximal.
        \pi(i, d) = argmax(\hat(\tau)(i, 1), ..., \hat(\tau)(i, \delta)) * b(i)

          1. For each entity i, pick exactly one dose d: the dose that yields the highest
             (CADE * b[i]) -- either from ground-truth or the model estimate.
          2. Sort all entities by this best yield (descending).
          3. Assign the chosen dose in that order, as long as total cost does not exceed 'budget'.
          4. If adding an entity would exceed the budget, assign it no treatment (dose=0).
          5. Return (objective, decision_vars) in the same format as solve_exact,
             (this ignores any other constraints like fainress).

        Parameters:
            budget (float): The budget constraint. Defaults to infinity.
        """

        # -------------------------------------------------
        # 1. Create decision_vars in the same "PuLP format"
        #    as your solve_exact, i.e., a dict { (i,j) : LpVariable }
        # -------------------------------------------------
        n = self.data.x_test.shape[0]
        d = self.ntreatmentlevels

        decision_vars = {
            (i, j): LpVariable(name=f"d{i}_{j}", lowBound=0, upBound=1, cat='Binary')
            for i in range(1, n + 1)
            for j in range(1, d + 1)
        }

        # -------------------------------------------------
        # 2. Build yield_matrix based on ground_truth flag
        # -------------------------------------------------

        # benefit parameters
        # b is a vector, redefine b as to be defined on a test set - use the indexing used in data.x_test:


        if ground_truth:
            yield_matrix = self.ite_gt_matrix * self.b_test.reshape(-1, 1)
            print(f'finding $\stackrel{budget}\Pi^{{opt}}$ with heuristic1 - using ground truth CADEs')
        else:
            yield_matrix = self.ite_est_matrix * self.b_test.reshape(-1, 1)
            print(f'finding $\stackrel{budget}\Pi^{{exp}}$ with heuristic1 - using estimated CADEs')

        # For each entity i, find the best dose j (max yield)
        import numpy as np
        best_dose_indices = np.argmax(yield_matrix, axis=1)  # best column j per row i
        best_yields = yield_matrix[np.arange(n), best_dose_indices]
        best_costs = np.array([self.c[i, best_dose_indices[i]] for i in range(n)])

        # -------------------------------------------------
        # 3. Sort entities by best yield (descending)
        # -------------------------------------------------
        entity_info = [(i, best_dose_indices[i], best_yields[i], best_costs[i]) for i in range(n)]
        entity_info.sort(key=lambda x: x[2], reverse=True)

        # -------------------------------------------------
        # 4. Greedily assign within budget
        #    We'll store the total objective manually.
        # -------------------------------------------------
        spent = 0.0
        objective = 0.0

        # By default, all varValue are 0:
        for (i, j) in decision_vars:
            decision_vars[(i, j)].varValue = 0

        # In sorted order, choose each best dose if budget allows
        for (i0, j_star, y_star, c_star) in entity_info:
            if spent + c_star <= budget:
                # i0 is 0-based, so the LpVariable index is (i0+1, j_star+1)
                decision_vars[(i0+1, j_star+1)].varValue = 1
                spent += c_star
                objective += y_star

        # -------------------------------------------------
        # 5. Attach a .value() method, so we can do
        #    `decision_vars[i, j].value()` in get_prescr_obj
        # -------------------------------------------------
        for var in decision_vars.values():
            # Attach a small function returning varValue
            var.value = lambda v=var: v.varValue

        return objective, decision_vars

    def solve_heuristic2(self, budget=float('inf'), ground_truth=False):
        """
            heuristic2: Selects the treatment with the highest payoff per unit cost.

            1. For each entity i, pick exactly one dose d: the dose that yields the highest
               (CADE * b[i] / cost) -- either from ground-truth or the model estimate.
            2. Sort all entities by this best yield per cost (descending).
            3. Assign the chosen dose in that order, as long as total cost does not exceed 'budget'.
            4. If adding an entity would exceed the budget, assign it no treatment (dose=0).
            5. Return (objective, decision_vars) in the same format as solve_exact,
               (this ignores any other constraints like fairness).

            Parameters:
                budget (float): The budget constraint. Defaults to infinity.
                ground_truth (bool): Whether to use ground truth data. Defaults to False.
            """

        # Initialize decision variables in the same "PuLP format" as solve_exact
        n = self.data.x_test.shape[0]
        d = self.ntreatmentlevels

        decision_vars = {
            (i, j): LpVariable(name=f"d{i}_{j}", lowBound=0, upBound=1, cat='Binary')
            for i in range(1, n + 1)
            for j in range(1, d + 1)
        }

        # Build yield_matrix based on ground_truth flag
        if ground_truth:
            yield_matrix = self.ite_gt_matrix * self.b_test.reshape(-1, 1)
            print(f'finding $\stackrel{{budget}}\Pi^{{opt}}$ with heuristic2 - using ground truth CADEs')
        else:
            yield_matrix = self.ite_est_matrix * self.b_test.reshape(-1, 1)
            print(f'finding $\stackrel{{budget}}\Pi^{{exp}}$ with heuristic2 - using estimated CADEs')

        # Calculate the payoff per unit cost
        payoff_per_cost = np.zeros_like(yield_matrix)
        for i in range(n):
            for j in range(d):
                cost = self.c[i, j]
                payoff_per_cost[i, j] = yield_matrix[i, j] / cost if cost > 0 else 0

        # For each entity i, find the best dose j (max payoff per cost)
        best_dose_indices = np.argmax(payoff_per_cost, axis=1)  # best column j per row i
        best_payoffs = payoff_per_cost[np.arange(n), best_dose_indices]
        best_costs = np.array([self.c[i, best_dose_indices[i]] for i in range(n)])

        # Sort entities by best payoff per cost (descending)
        entity_info = [(i, best_dose_indices[i], best_payoffs[i], best_costs[i]) for i in range(n)]
        entity_info.sort(key=lambda x: x[2], reverse=True)

        # Greedily assign within budget
        spent = 0.0
        objective = 0.0

        # By default, all varValue are 0:
        for (i, j) in decision_vars:
            decision_vars[(i, j)].varValue = 0

        # In sorted order, choose each best dose if budget allows
        for (i0, j_star, payoff_star, cost_star) in entity_info:
            if spent + cost_star <= budget:
                # i0 is 0-based, so the LpVariable index is (i0+1, j_star+1)
                decision_vars[(i0 + 1, j_star + 1)].varValue = 1
                spent += cost_star
                objective += payoff_star

        # Attach a .value() method, so we can do `decision_vars[i, j].value()` in get_prescr_obj
        for var in decision_vars.values():
            var.value = lambda v=var: v.varValue

        return objective, decision_vars

    def get_prescr_obj(self, decision_vars):
        """
        Computes the prescriptive objective value.

        Parameters:
            decision_vars: The decision variables from the optimization.

        Returns:
            float: The prescriptive objective value.
        """
        obj = 0
        for i, j in decision_vars.keys():
            obj += decision_vars[i, j].value() * self.ite_gt_matrix[i - 1, j - 1] * self.b[i - 1]

        return obj

    def get_prescr_objU(self, decision_vars):
        """
        Computes the prescriptive objective value for uniform benefit.

        Parameters:
            decision_vars: The decision variables from the optimization.

        Returns:
            float: The prescriptive objective value for uniform benefit.
        """
        num_rows = self.data.x_test.shape[0]
        b = np.ones(num_rows)
        objU = 0
        for i, j in decision_vars.keys():
            objU += decision_vars[i, j].value() * self.ite_gt_matrix[i - 1, j - 1] * b[i - 1]

        return objU

    def get_prescr_objV(self, decision_vars):
        """
        Computes the prescriptive objective value for benefit-dependent version.

        Parameters:
            decision_vars: The decision variables from the optimization.

        Returns:
            float: The prescriptive objective value for benefit-dependent version.
        """
        b = self.data.b  # benefit-dependent version
        objV = 0
        for i, j in decision_vars.keys():
            objV += decision_vars[i, j].value() * self.ite_gt_matrix[i - 1, j - 1] * b[i - 1]

        return objV

def find_optimal_solution(data, model, ntreatments, optimization_settings, budget):
    """
    Finds the optimal full-information solution (V^opt).

    Parameters:
        data: The dataset containing the features and outcomes.
        model: The predictive model used for estimating treatment effects.
        ntreatments: The number of treatment levels.
        optimization_settings: Settings for the optimization process.
        budget: The budget constraint.

    Returns:
        float: The optimal objective value.
    """
    problem = Optimization_problem(data=data, model=model, ntreatmentlevels=ntreatments,
                                   optimization_settings=optimization_settings)
    obj_opt, _ = problem.solve(method="exact", budget=budget, ground_truth=True)

    return obj_opt

def find_obj_exp(data, model, ntreatments, optimization_settings, budget):
    """
    Finds the prescriptive solution and expected value (V^exp).

    Parameters:
        data: The dataset containing the features and outcomes.
        model: The predictive model used for estimating treatment effects.
        ntreatments: The number of treatment levels.
        optimization_settings: Settings for the optimization process.
        budget: The budget constraint.

    Returns:
        tuple: The expected objective value, decision variables, and the optimization problem instance.
    """
    problem = Optimization_problem(data=data, model=model, ntreatmentlevels=ntreatments,
                                   optimization_settings=optimization_settings)
    obj_exp, decision_vars = problem.solve(method=optimization_settings['exact'], budget=budget, ground_truth=False)

    return obj_exp, decision_vars, problem

def find_true_objective_value(problem, decision_vars):
    """
    Finds the true objective value of the prescriptive solution (V^est).

    Parameters:
        problem: The optimization problem instance.
        decision_vars: The decision variables from the optimization.

    Returns:
        float: The true objective value.
    """
    obj_prescr = problem.get_prescr_objective_value(decision_vars)

    return obj_prescr

def calculate_regret(obj_opt, obj_prescr):
    """
    Calculates the regret and normalized regret.

    Parameters:
        obj_opt: The optimal objective value.
        obj_prescr: The prescriptive objective value.

    Returns:
        tuple: The regret and normalized regret.
    """
    regret = obj_opt - obj_prescr
    regret_norm = regret / obj_opt

    return regret, regret_norm

def get_budget_used(decision_vars, c):
    """
    Calculates the budget used based on the decision variables.

    Parameters:
        decision_vars: The decision variables from the optimization.
        c: The cost parameters.

    Returns:
        float: The budget used.
    """
    budget_used = 0.0
    for i, j in decision_vars.keys():
        budget_used += decision_vars[i, j].value() * c[i - 1, j - 1]

    return budget_used


def get_fairness_info(decision_vars, problem, data, A):
    """
    Calculates fairness information based on the decision variables.

    Parameters:
        decision_vars: The decision variables from the optimization.
        problem: The optimization problem instance.
        data: The dataset containing the features and outcomes.
        A: The protected attribute.

    Returns:
        tuple: Fairness metrics including total doses, average doses, and ratios for protected and non-protected groups.
    """
    total_doses_prot_0 = 0
    total_doses_prot_1 = 0
    total_estite_prot_0 = 0
    total_estite_prot_1 = 0
    total_gtite_prot_0 = 0
    total_gtite_prot_1 = 0

    for i, j in decision_vars.keys():
        if data.x_test[i - 1, A] == 0:
            total_doses_prot_0 += decision_vars[i, j].value() * problem.s[i - 1, j - 1]
            total_estite_prot_0 += decision_vars[i, j].value() * problem.ite_est_matrix[
                i - 1, j - 1]
            total_gtite_prot_0 += decision_vars[i, j].value() * problem.ite_gt_matrix[i - 1, j - 1]
        else:  # if A==1:
            total_doses_prot_1 += decision_vars[i, j].value() * problem.s[i - 1, j - 1]
            total_estite_prot_1 += decision_vars[i, j].value() * problem.ite_est_matrix[
                i - 1, j - 1]
            total_gtite_prot_1 += decision_vars[i, j].value() * problem.ite_gt_matrix[i - 1, j - 1]

    avg_dose_prot_0 = total_doses_prot_0 / np.sum(data.x_test[:, A] == 0)
    avg_dose_prot_1 = total_doses_prot_1 / np.sum(data.x_test[:, A] == 1)

    avg_estite_prot_0 = total_estite_prot_0 / np.sum(data.x_test[:, A] == 0)
    avg_estite_prot_1 = total_estite_prot_1 / np.sum(data.x_test[:, A] == 1)

    avg_gtite_prot_0 = total_gtite_prot_0 / np.sum(data.x_test[:, A] == 0)
    avg_gtite_prot_1 = total_gtite_prot_1 / np.sum(data.x_test[:, A] == 1)

    ratio_dose_prot = avg_dose_prot_0 / avg_dose_prot_1
    ratio_estite_prot = avg_estite_prot_0 / avg_estite_prot_1
    ratio_gtite_prot = avg_gtite_prot_0 / avg_gtite_prot_1

    return total_doses_prot_0, total_doses_prot_1, avg_dose_prot_0, avg_dose_prot_1, avg_estite_prot_0, avg_estite_prot_1, avg_gtite_prot_0, avg_gtite_prot_1, ratio_dose_prot, ratio_estite_prot, ratio_gtite_prot


def optimize_and_track_progress(data, model, model_settings, results_predict, RES_OPT_FILE, optimization_settings_list):
    """
    Optimizes the treatment allocation and tracks the progress.

    Parameters:
        data: The dataset containing the features and outcomes.
        model: The predictive model used for estimating treatment effects.
        model_settings: Settings for the model.
        results_predict: The results from the prediction phase.
        RES_OPT_FILE: The file to store optimization results.
        optimization_settings_list: List of optimization settings.

    Returns:
        dict: The optimization results.
    """
    combinations = list(itertools.product(*optimization_settings_list.values()))

    print(f"Optimization for '{model_settings['model_name']}'  started. {len(combinations)} runs."
          f"\n{optimization_settings_list}")

    results_optimize = results_predict.copy()
    pbar = tqdm(total=len(combinations))
    run = 1

    for combination in combinations:
        exact, cost_sensitive_optimization, protected_attribute, epsilon_disparate_treatment, epsilon_disparate_outcome, budget, ntreatments = combination

        optimization_settings = {
            "exact": exact,
            "cost_sensitive_optimization": cost_sensitive_optimization,
            "protected_attribute": protected_attribute,
            "epsilon_disparate_treatment": epsilon_disparate_treatment,
            "epsilon_disparate_outcome": epsilon_disparate_outcome,
            "budget": budget,
            "ntreatments": ntreatments
        }

        optimization_settings_U = optimization_settings.copy()
        optimization_settings_U["cost_sensitive_optimization"] = False

        optimization_settings_V = optimization_settings.copy()
        optimization_settings_V["cost_sensitive_optimization"] = True



        # Get optimal, expected, and prescriptive solutions
        obj_opt = find_optimal_solution(data, model, ntreatments, optimization_settings, budget)
        objU_opt = find_optimal_solution(data, model, ntreatments, optimization_settings_U, budget)
        objV_opt = find_optimal_solution(data, model, ntreatments, optimization_settings_V, budget)

        start_time = time.time()
        obj_exp, decision_vars, problem = find_obj_exp(data, model, ntreatments, optimization_settings, budget)
        time_calc = time.time() - start_time

        obj_prescr = problem.get_prescr_obj(decision_vars=decision_vars)
        objU_prescr = problem.get_prescr_objU(decision_vars=decision_vars)
        objV_prescr = problem.get_prescr_objV(decision_vars=decision_vars)

        # Calculate regret
        regret, regret_norm = calculate_regret(obj_opt, obj_prescr)
        regretU, regretU_norm = calculate_regret(objU_opt, objU_prescr)
        regretV, regretV_norm = calculate_regret(objV_opt, objV_prescr)



        # Constraint information:
        budget_used = get_budget_used(decision_vars, problem.c)
        total_doses_prot_0, total_doses_prot_1, avg_dose_prot_0, avg_dose_prot_1, avg_estite_prot_0, avg_estite_prot_1, avg_gtite_prot_0, avg_gtite_prot_1, ratio_dose_prot, ratio_estite_prot, ratio_gtite_prot = get_fairness_info(
            decision_vars=decision_vars, problem=problem, data=data, A=optimization_settings["protected_attribute"])

        results_optimize.update({
            "model": model_settings["model_name"],
            "model_type": model_settings["model_type"],

            "exact": exact,
            "cost_sensitive_optimization": cost_sensitive_optimization,
            "protected_attribute": protected_attribute,
            "epsilon_disparate_treatment": epsilon_disparate_treatment,
            "epsilon_disparate_outcome": epsilon_disparate_outcome,
            "budget": budget,
            "ntreatments": ntreatments,

            "budget": budget,
            "ntreatments": ntreatments,
            "obj_opt": obj_opt,
            "objU_opt": objU_opt,
            "objV_opt": objV_opt,
            "obj_exp": obj_exp,
            "obj_prescr": obj_prescr,
            "objU_prescr": objU_prescr,
            "objV_prescr": objV_prescr,
            "regret": regret,
            "regretU": regretU,
            "regretV": regretV,
            "regret_norm": regret_norm,
            "regretU_norm": regretU_norm,
            "regretV_norm": regretV_norm,

            "budget_used": budget_used,

            "protected_attribute": optimization_settings["protected_attribute"],
            "total_doses_prot_0": total_doses_prot_0,
            "total_doses_prot_1": total_doses_prot_1,
            "avg_dose_prot_0": avg_dose_prot_0,
            "avg_dose_prot_1": avg_dose_prot_1,
            "ratio_dose_prot": ratio_dose_prot,
            "avg_estite_prot_0": avg_estite_prot_0,
            "avg_estite_prot_1": avg_estite_prot_1,
            "ratio_estite_prot": ratio_estite_prot,
            "avg_gtite_prot_0": avg_gtite_prot_0,
            "avg_gtite_prot_1": avg_gtite_prot_1,
            "ratio_gtite_prot": ratio_gtite_prot,

            "time_calc": round(time_calc, 4)
        })

        add_dict(RES_OPT_FILE, results_optimize)

        pbar.update(1)
        run += 1

    return results_optimize, decision_vars
