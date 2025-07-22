from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
import numpy as np
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from predict_general import general_predict
from pymoo.operators.sampling.rnd import IntegerRandomSampling

class ImageAttackProblem(Problem):
    def __init__(self, n_pixels):
        super().__init__(n_var = n_pixels * 2, n_obj = 1, n_constr = 0, xl = 0, xu = 31)

        self.n_pixels = n_pixels


    def update_parameters(self, image,label,model,model_type):
        self.image = image
        self.label = label
        self.model = model
        self.model_type = model_type
        
    def _evaluate(self, x, out, *args, **kwargs):
        
        adv_predictions = []
        for idx in range(x.shape[0]):
            
            pixels = np.reshape(x[idx], (self.n_pixels, -1))
            x_adv = self.image.copy()
            for i in range(self.n_pixels):
                x_adv[round(pixels[i][0]), round(pixels[i][1])] = 0

            adv_prediction = general_predict(self.model, x_adv, self.model_type)

            if np.argmax(adv_prediction) != np.argmax(self.label):
                adv_predictions.append(-max(adv_prediction))
            else:
                adv_predictions.append(1000)
            
            
            if adv_predictions[-1] < -0.7 :
                for j in range(idx,x.shape[0]):
                    adv_predictions.append(1000)
                break
            
        out["F"]=[adv_predictions]


class Attack():
    def __init__(self, n_pixels, popsize, maxiter):
        self.problem = ImageAttackProblem(n_pixels)
        self.algorithm = DE(pop_size=popsize,sampling=IntegerRandomSampling())
        self.maxiter = maxiter
        self.n_pixels = n_pixels
        
        self.termination = DefaultMultiObjectiveTermination(
            xtol=1e-8,
            cvtol=1e-7,
            ftol=0.002,
            period=30,
            n_max_gen=maxiter,
            n_max_evals=100000
        )
        
        
    def attack_100_pixel(self,image, label, model, model_type):
        
        self.problem.update_parameters(image,label,model,model_type)
        res = minimize(self.problem, self.algorithm, termination=self.termination)

        best_solution = res.X
        best_solution = np.reshape(best_solution, (self.n_pixels, -1))

        x_adv = image.copy()
        for i in range(self.n_pixels):
            x_adv[round(best_solution[i][0]), round(best_solution[i][1])] = 0

        return x_adv, res.F[0], self.maxiter
