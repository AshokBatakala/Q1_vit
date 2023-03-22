from torch.utils.data import DataLoader
import torch 
import matplotlib.pyplot as plt
import os
class Trainer:
    """ Trainer class for training and evaluating models
    usage:
    trainer = Trainer(model,optimizer,criterion,train_set = trainset,val_set = valset,test_set = testset)
    trainer.train(epochs=10)
    trainer.plot() # plot training and validation metrics
    trainer.test() # it returns the log
    """
    def __init__(self,model,optimizer,criterion,train_set = None,val_set = None,test_set = None,device = None,batch_size = 32,shuffle=True,num_workers=2):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        # log 
        self.last_log = None

    def create_data_loader(self,dataset):
        return DataLoader(dataset,batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers)

        
        
    def test(self,test_set = None):
        """ test model on test set
        Args:
            test_set (torch.utils.data.Dataset): test set
        Returns:
            log (dict): log dict
        """
        if test_set is None:
            test_set = self.test_set
        
        # create loader
        test_loader = self.create_data_loader(test_set)
        
        # log dict
        log = {'test_loss':[],'test_acc':[]}

        device = self.device
        
        #model to eval mode
        model = self.model
        model = model.to(device)
        model.eval()
        
        epoch_loss = 0
        epoch_corrects = 0
        for i, data in enumerate(test_loader): # for each batch
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = self.criterion(outputs, labels)
            
            # statistics
            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            epoch_corrects += torch.sum(preds == labels.data)
                
        # calculate loss and accuracy
        epoch_loss = epoch_loss / len(test_set)
        epoch_acc = epoch_corrects.double() / len(test_set)
        
        # add to log
        log['test_loss'].append(epoch_loss)
        log['test_acc'].append(epoch_acc)
        
        return log

    def train(self,n_epochs = 10,return_log = False):
        """ training function
        
        Args:
            n_epochs (int): number of epochs
            return_log (bool): if True return log dict
        Returns:
            log (dict): log dict

        Note: loss  devided by number of data points
        """


        # create loaders
        train_loader = self.create_data_loader(self.train_set)
        val_loader = self.create_data_loader(self.val_set)
        
        # setting variables
        device = self.device
        model = self.model
        model = model.to(device)
        optimizer = self.optimizer
        criterion = self.criterion



        # log dict
        log = {'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[]}
        

        for epoch in range(n_epochs):
            
            epoch_loss = 0
            epoch_corrects = 0
            for i, data in enumerate(train_loader): # for each batch
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                
                outputs = model(inputs)
                loss = criterion(outputs,labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_corrects += torch.sum(torch.argmax(outputs,1) == labels.data)

            epoch_loss = epoch_loss/len(train_loader.dataset)
            epoch_acc = epoch_corrects.double()/len(train_loader.dataset)
            log['train_loss'].append(epoch_loss)
            log['train_acc'].append(epoch_acc)

            # validation using test function
            if self.val_set is not None:
                val_log = self.test(self.val_set)
                log['val_loss'].append(val_log['test_loss'][0])
                log['val_acc'].append(val_log['test_acc'][0])
        
        # store last_train_log 
        self.last_log = log
        if return_log:
            return log


    def plot(self,log = None,figsize = (10,5),save_path = None,title = None,show = True,filename = 'plot.png'):
        """ saving : it uses savepath + filename
        so save_path should end with '/'
        """ 
        if log is None:
            log = self.last_log

        plt.figure(figsize= figsize)
        plt.subplot(1,2,1)
        plt.plot(log['train_loss'],label='train_loss')
        plt.plot(log['val_loss'],label='val_loss')
        # x and y labels
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # title
        if title is not None:
            plt.title(title)        
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(log['train_acc'],label='train_acc')
        plt.plot(log['val_acc'],label='val_acc')
        plt.legend()
        # x and y labels
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        # title
        if title is not None:
            plt.title(title)
        if save_path is not None:
            plt.savefig(save_path + filename)
        if show:
            plt.show()
        
    def save_model(self,save_path = "model_zoo/",filename = 'model.pth'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.model.state_dict(),save_path + filename)

    def load_model(self,load_path = "model_zoo/",filename = 'model.pth'):
       
        # check file exists
        if os.path.isfile(load_path + filename):
            try:
                self.model.load_state_dict(torch.load(load_path + filename))
            except:
                raise ValueError('File not compatible')
        else:
            raise ValueError('File not found')
            



