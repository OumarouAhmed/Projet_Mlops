python -m src.train --model logreg --C 0.5 --solver lbfgs --run_name variation_lr
python -m src.train --model svm --C 1.0 --kernel rbf --run_name variation_svm
