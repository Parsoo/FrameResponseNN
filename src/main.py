import generate_data
import preprocess
import train
import evaluate

def main():
    print("Step 1: Generating synthetic data...")
    generate_data.main(n_samples=500)   # you can change n_samples if time is short

    print("Step 2: Preprocessing...")
    preprocess.preprocess()

    print("Step 3: Training MLP...")
    train.train()

    print("Step 4: Evaluating...")
    evaluate.evaluate()

    print("All done! Check results/ for plots and models/ for saved model.")

if __name__ == '__main__':
    main()