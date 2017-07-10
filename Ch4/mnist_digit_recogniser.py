import mlp
import idx2numpy
from numpy import shape, ndarray, transpose, reshape, concatenate, ones, zeros,\
    where, float128, nan, argmax

class mnist_digit_recogniser:
    def __init__(self, path_to_mnist_train_dataset, path_to_mnist_train_label_set, path_to_mnist_test_dataset, path_to_mnist_test_label_set):
        # Gathering all training data first
        self.train_data_path = path_to_mnist_train_dataset
        self.train_label_path = path_to_mnist_train_label_set
        
        self.train_array = idx2numpy.convert_from_file(path_to_mnist_train_dataset)
        self.train_label_array = idx2numpy.convert_from_file(path_to_mnist_train_label_set)

        #self.train_array = self.train_array[:5000,:]
        #self.train_label_array = self.train_label_array[:5000]

        print 'Original Training Array Size = ', shape(self.train_array)
        print 'Original Training Label Array Size = ', shape(self.train_label_array)

        self.train_array = reshape(self.train_array, (60000, 784))
        self.train_label_array = reshape(self.train_label_array, (60000, 1))
        
        self.train_array.flags.writeable = True
        self.train_array = self.train_array.astype(float128)
        self.train_array[:,:] = (self.train_array[:,:] - self.train_array[:,:].mean(axis = 0)) / 255.0
        
        # Gathering all the test data
        self.test_data_path = path_to_mnist_test_dataset
        self.test_array = idx2numpy.convert_from_file(path_to_mnist_test_dataset)
        self.test_label_path = path_to_mnist_test_label_set
        self.test_label_array = idx2numpy.convert_from_file(path_to_mnist_test_label_set)

        self.test_array.flags.writeable = True
        
        self.test_array = reshape(self.test_array, (10000, 784))
        self.test_label_array = reshape(self.test_label_array, (10000, 1))
        self.test_array = self.test_array.astype(float128)
        self.test_array[:,:] = (self.test_array[:,:] - self.test_array[:,:].mean(axis = 0)) / 255.0

        self.target_1_of_n = zeros((shape(self.train_label_array)[0], 10))
        indices = where(self.train_label_array[:, 0] == 0)
        self.target_1_of_n[indices, 0] = 1;
        indices = where(self.train_label_array[:, 0] == 1)
        self.target_1_of_n[indices, 1] = 1;
        indices = where(self.train_label_array[:, 0] == 2)
        self.target_1_of_n[indices, 2] = 1;
        indices = where(self.train_label_array[:, 0] == 3)
        self.target_1_of_n[indices, 3] = 1;
        indices = where(self.train_label_array[:, 0] == 4)
        self.target_1_of_n[indices, 4] = 1;
        indices = where(self.train_label_array[:, 0] == 5)
        self.target_1_of_n[indices, 5] = 1;
        indices = where(self.train_label_array[:, 0] == 6)
        self.target_1_of_n[indices, 6] = 1;
        indices = where(self.train_label_array[:, 0] == 7)
        self.target_1_of_n[indices, 7] = 1;
        indices = where(self.train_label_array[:, 0] == 8)
        self.target_1_of_n[indices, 8] = 1;
        indices = where(self.train_label_array[:, 0] == 9)
        self.target_1_of_n[indices, 9] = 1;
        
        self.test_1_of_n = zeros((shape(self.test_label_array)[0], 10))
        indices = where(self.test_label_array[:, 0] == 0)
        self.test_1_of_n[indices, 0] = 1;
        indices = where(self.test_label_array[:, 0] == 1)
        self.test_1_of_n[indices, 1] = 1;
        indices = where(self.test_label_array[:, 0] == 2)
        self.test_1_of_n[indices, 2] = 1;
        indices = where(self.test_label_array[:, 0] == 3)
        self.test_1_of_n[indices, 3] = 1;
        indices = where(self.test_label_array[:, 0] == 4)
        self.test_1_of_n[indices, 4] = 1;
        indices = where(self.test_label_array[:, 0] == 5)
        self.test_1_of_n[indices, 5] = 1;
        indices = where(self.test_label_array[:, 0] == 6)
        self.test_1_of_n[indices, 6] = 1;
        indices = where(self.test_label_array[:, 0] == 7)
        self.test_1_of_n[indices, 7] = 1;
        indices = where(self.test_label_array[:, 0] == 8)
        self.test_1_of_n[indices, 8] = 1;
        indices = where(self.test_label_array[:, 0] == 9)
        self.test_1_of_n[indices, 9] = 1;
        
        self.m = mlp.mlp(self.train_array.astype(float128), self.target_1_of_n.astype(float128), 4, outtype='softmax')

    def createModel(self):
        self.m.mlptrain(self.train_array.astype(float128), self.target_1_of_n.astype(float128), 0.25, 100)
        self.m.save()

    def testModel(self):
        print 'Testing Model ..'
        #import ipdb;
        #ipdb.set_trace()
        correct = 0
        incorrect = 0
        idx = 0
        import csv
        with open('outputs.csv', 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in self.test_array:
                i = reshape(i, (1, 784))
                i = i.astype(float128)
                i = concatenate((i,-ones((1,1))),axis=1)
                output = self.m.mlpfwd2(i)
                detected = argmax(output[0])
                print 'Actual = ', self.test_label_array[idx,0]
                print 'Detected = ', detected
                spamwriter.writerow([self.test_label_array[idx,0], detected, output])
                #print 'Output = ', output[0]
                
                if (detected == self.test_label_array[idx,0]):
                    correct += 1;
                else:
                    incorrect += 1;
                
                idx += 1;
        
        csvfile.close()
                
        print 'Correct = ', correct, ' Incorrect = ', incorrect
        
        print '---> Confusion Matrix ='
        #    print 'Evaluated Output = ', output
        self.m.confmat(self.test_array,self.test_1_of_n)

recogniser = mnist_digit_recogniser('/home/ubuntu/ml/data/train-images-idx3-ubyte', '/home/ubuntu/ml/data/train-labels-idx1-ubyte', 
                                    '/home/ubuntu/ml/data/t10k-images-idx3-ubyte', '/home/ubuntu/ml/data/t10k-labels-idx1-ubyte')

print 'Training MLP Model for Logistic output ..'
recogniser.createModel()
recogniser.testModel()