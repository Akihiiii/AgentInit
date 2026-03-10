class Accuracy:
    def __init__(self):
        self._num_correct = 0
        self._num_total = 0
        self._last_record = None  # 存储最近一次对比

    
    def update(self, predicted: str, target: str) -> None:
        is_correct = predicted == target
        self._num_correct += int(is_correct)
        self._num_total += 1
        self._last_record = {
            'predicted': predicted,
            'target': target,
            'is_correct': is_correct
        }

    def get(self) -> float:
        return self._num_correct / self._num_total

    def print(self):
        accuracy = self.get()
        print(f"Accuracy: {accuracy*100:.1f}% "
              f"({self._num_correct}/{self._num_total})")
        if self._last_record:
            status = "✓" if self._last_record['is_correct'] else "✗"
            print(f"  Current: [{status}] Pred: {self._last_record['predicted']!r} "
                  f"| Target: {self._last_record['target']!r}")