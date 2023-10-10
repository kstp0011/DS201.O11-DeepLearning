import torch

# Tính độ chính xác, độ nhạy và độ đặc hiệu cho từng chữ số

classwise_correct = []
classwise_total = []
for i in range(10):
    classwise_correct.append(0)
    classwise_total.append(0)

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for j in range(len(labels)):
            if labels[j] == i:
                classwise_total[i] += 1
                if predicted[j] == labels[j]:
                    classwise_correct[i] += 1

# Tính độ chính xác, độ nhạy và độ đặc hiệu cho từng chữ số

for i in range(10):
    if classwise_total[i] == 0:
        continue
    accuracy = classwise_correct[i] / classwise_total[i]
    precision = classwise_correct[i] / (classwise_correct[i] + (classwise_total[i] - classwise_correct[i]))
    recall = classwise_correct[i] / (classwise_correct[i] + (classwise_total[i] - classwise_correct[i]))
    print('Class: ', i)
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('-----------------------')
    
