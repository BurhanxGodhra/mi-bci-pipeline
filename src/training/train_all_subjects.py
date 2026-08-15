"""
Step 8-fix: Retrain all 9 subjects with corrected (leak-free) methodology.
"""
import numpy as np
from train import train_subject


def main():
    results = []
    for subject_id in range(1, 10):
        result = train_subject(subject_id=subject_id, n_epochs=100, verbose=True)
        results.append(result)

    final_accs = [r["final_test_acc"] for r in results]

    print(f"\n{'='*70}")
    print("MULTI-SUBJECT SUMMARY (corrected methodology -- test set touched once)")
    print(f"{'='*70}")
    print(f"{'Subject':<10}{'InternalVal':<14}{'FinalTestAcc':<14}{'TrainN':<10}{'ValN':<8}{'TestN':<8}")
    for r in results:
        print(f"{r['subject_id']:<10}{r['best_internal_val_acc']:<14.3f}{r['final_test_acc']:<14.3f}"
              f"{r['n_train']:<10}{r['n_val']:<8}{r['n_test']:<8}")

    print(f"\nMean FINAL TEST accuracy across subjects: {np.mean(final_accs):.3f}")
    print(f"Std deviation: {np.std(final_accs):.3f}")
    print(f"Best subject: {results[np.argmax(final_accs)]['subject_id']} ({max(final_accs):.3f})")
    print(f"Worst subject: {results[np.argmin(final_accs)]['subject_id']} ({min(final_accs):.3f})")


if __name__ == "__main__":
    main()