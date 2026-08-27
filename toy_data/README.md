# Experiments on toy data

[Report/plots] (https://docs.google.com/document/d/15C3-MsQtY3VW3y6I7jK_BtYqUy_Q3QtMyfwyfLsWD00/edit?tab=t.0)

## Heatmap generation (PVOROS score vs. angle/intercept 2D params)
- Run:
  ```
  bash w1w2_sweep.sh (sigmoid)      # computes (soft) pVOROS score across grid of parameters
  python make_plot.py --filepath    # creates heatmap plot
  python 1d_plot.py                 # plots pvoros score against a fixed weight
  ```               

## Gradient descent
- Run:
  ```
  python gradient_descent.py
  ```
- Performs gradient descent using soft PVOROS loss for each of the 5 seeds
- 10 random initializations
- plots all trajectories on previously made heatmap, best convergence highlighted


## BCE vs. PV
- Run:
  ```
  python bce_vs_pv.py
  ```

- Compare results of 3 training methods:
  1) min BCE
  2) min BCE while searching best train PV along trajectory
  3) min soft PV loss
- Then evaluate with chosen parameters on test set
