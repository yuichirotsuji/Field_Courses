# import necessary packages
# (Note: The pyblp doesn't work with numpy 2.X due to numpy.unicode configulartion in it.
#  Hence we have to downgrade it to numpy 1.26.4 (which I did before running this code.))
import numpy as np
import pandas as pd 
import pyblp

# problem formulation using Nevo's data (As in Colon and Gortmaker(2020))
problem = pyblp.Problem(
    product_formulations=(
        pyblp.Formulation('0 + prices', absorb = 'C(product_ids)'),
        pyblp.Formulation('1 + prices + sugar + mushy'),
    ),
    agent_formulation = pyblp.Formulation('0 + income + income_squared + age + child'),
    product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION),
    agent_data = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)
)

# problem solving with intial values used by Nevo (As in Colon and Gortmaker(2020))
results = problem.solve(
    sigma = np.diag([0.3302, 2.4526, 0.0163, 0.2241]),
    pi = [
        [ 5.4819, 0,      0.2037, 0      ],
        [15.8935, -1.200, 0     , 2.63442],
        [-0.2506, 0,      0.0511, 0      ],
        [ 1.2650, 0,     -0.8901, 0      ]
    ],
    method = '1s',
    optimization = pyblp.Optimization('bfgs', {'gtol': 1e-3})
)

print(results) # get the estimation results

elasticities = results.compute_elasticities()
print(elasticities)
markups = results.compute_markups()
print(markups)
