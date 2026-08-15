import numpy as np
from train import train_subject_refit

def main():
    results = [train_subject_refit(s, verbose=True) for s in range(1, 10)]
    accs = [r["final_test_acc"] for r in results]
    print(f"\nMean: {np.mean(accs):.3f} | Std: {np.std(accs):.3f}")
    for r in results:
        print(f"Subject {r['subject_id']}: {r['final_test_acc']:.3f} (stopped epoch {r['stopping_epoch']})")

if __name__ == "__main__":
    main()