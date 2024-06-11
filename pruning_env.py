from libs import *
from evaluate import evaluate_model

class PruningEnv:
    def __init__(self, model, test_loader, pruning_percentage=0.2):
        self.model = model
        self.test_loader = test_loader
        self.pruning_percentage = pruning_percentage
        self.original_accuracy = self.evaluate()
    
    def evaluate(self):
        return evaluate_model(self.model, self.test_loader)
    
    def prune(self):
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                mask = torch.rand(param.data.size()) > self.pruning_percentage
                param.data *= mask.float()
    
    def reset(self):
        self.model.apply(self.weights_init)
        return self.evaluate()
    
    def step(self, action):
        self.prune()    
        new_accuracy = self.evaluate()
        reward = new_accuracy - self.original_accuracy
        self.original_accuracy = new_accuracy
        return reward, new_accuracy
    
    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
