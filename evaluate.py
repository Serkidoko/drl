from libs import *

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_evaluation(model, train_loader, test_loader):
    # Đánh giá trên tập huấn luyện
    train_accuracy = evaluate_model(model, train_loader)
    print(f'Accuracy on training set: {train_accuracy:.2f}%')
    
    # Đánh giá trên tập kiểm tra
    test_accuracy = evaluate_model(model, test_loader)
    print(f'Accuracy on test set: {test_accuracy:.2f}%')
    
    # Số lượng tham số
    num_parameters = count_parameters(model)
    print(f'Number of parameters: {num_parameters}')




