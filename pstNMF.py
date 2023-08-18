import math
import random
import warnings
import numpy as np
import pandas as pd
import spams
import shutil
import csv
import os
import matplotlib.pyplot as plt

class staNMF:

    def __init__(self, filepath=None, data=None, folderID="", K1=2, K2=4, sample_weights=False, seed=123, replicates=100, NMF_finished=False, parallel=False, user_guess=None):
        warnings.filterwarnings("ignore")
        self.K1 = K1
        self.K2 = K2
        self.sample_weights = sample_weights
        self.seed = seed
        self.user_guess = user_guess  # User-provided guess matrix as np array
        self.guessdict = {}
        self.parallel = parallel
        self.replicates = range(replicates) if isinstance(replicates, int) else range(replicates[0], replicates[1])
        self.X = []
        self.data = data  # New attribute to store the provided numpy array
        self.fn = filepath  # Directly using the provided file path
        self.folderID = folderID
        self.rowidmatrix = []
        self.NMF_finished = NMF_finished
        self.instabilitydict = {}
        self.load_data()
        self.instabilityarray = []
        self.stability_finished = False
        random.seed(self.seed)

    def initialguess(self, X, K, i):
        if self.user_guess is not None:
            # If user provides a guess matrix, use it directly
            self.guess = np.asfortranarray(self.user_guess)
        else:
            # Otherwise, generate a random guess matrix as before
            indexlist = random.sample(range(1, X.shape[1]), K)
            self.guess = np.asfortranarray(X[:, indexlist])
            self.guessdict[i] = indexlist

    def load_data(self):
        if not self.NMF_finished:
            # Check if the 'data' attribute is provided
            if self.data is not None:
                workingmatrix = pd.DataFrame(self.data)
                self.rowidmatrix = workingmatrix.index.values
            else:
                workingmatrix = pd.read_csv(self.fn, index_col=0)
                self.rowidmatrix = workingmatrix.index.values
            
            if self.sample_weights:
                if isinstance(self.sample_weights, list) and len(self.sample_weights) != len(workingmatrix.columns):
                    raise ValueError("sample_weights length must equal the number of columns.")
                else:
                    colUnique, counts = np.unique(workingmatrix.columns, return_counts=True)
                    weight = 1.0 / counts
                    workingmatrix = workingmatrix.apply(lambda x: weight * x, axis=1)
                    workingmatrix = workingmatrix.applymap(math.sqrt)

            self.X = np.asfortranarray(workingmatrix.values.astype(float))

    def _nmf_worker(self, args):
        K, l, kwargs = args
        path = os.path.join("./staNMFDicts", str(self.folderID), f"K={K}")
        
        self.initialguess(self.X, K, l)
        Dsolution = spams.trainDL(self.X, D=self.guess, **kwargs)
        
        # Save the solution to a csv file
        outputfilename = f"factorization_{l}.csv"
        outputfilepath = os.path.join(path, outputfilename)
        Dsolution1 = pd.DataFrame(Dsolution, index=self.rowidmatrix)
        Dsolution1.to_csv(outputfilepath, header=None)
        
    def runNMF(self, **kwargs):
        self.NMF_finished = False
        numPatterns = np.arange(self.K1, self.K2+1)

        # Default parameters
        m, n = np.shape(self.X)
        default_params = {
            "numThreads": -1,
            "batchsize": min(1024, n),
            "lambda1": 0,
            "iter": 500,
            "mode": 2,
            "modeD": 0,
            "posAlpha": True,
            "posD": True,
            "verbose": False,
            "gamma1": 0
        }
        kwargs = {**default_params, **kwargs}  # Overwrite defaults with any provided kwargs

        for K in numPatterns:
            path = os.path.join("./staNMFDicts", str(self.folderID), f"K={K}")
            os.makedirs(path, exist_ok=True)

            # Sequentially compute NMF for each replicate
            for l in self.replicates:
                self._nmf_worker((K, l, kwargs))

            # Write the guess dictionary
            indexoutputpath = os.path.join(path, f"selectedcolumns{K}.csv")
            with open(indexoutputpath, "w") as indexoutputfile:
                for m, indices in sorted(self.guessdict.items()):
                    indexoutputfile.write(f"{m}\t{indices}\n")

        self.NMF_finished = True

    
    def amariMaxError(self, correlation):
        maxCol = np.absolute(correlation).max(0)
        maxRow = np.absolute(correlation).max(1)
        return (np.mean(1 - maxCol) + np.mean(1 - maxRow)) / 2

    def findcorrelation(self, A, B, k):
        return np.array([
            np.corrcoef(A[:, a], B[:, b])[0, 1]
            for a in range(k) for b in range(k)
        ]).reshape(k, k)

    def _instability_worker(self, args):
        i, Dhat, k = args
        distMat = np.zeros(len(self.replicates))
        for j in range(i, len(self.replicates)):
            CORR = self.findcorrelation(Dhat[i], Dhat[j], k)
            distMat[j] = self.amariMaxError(CORR)
        return distMat[i:]

    def instability(self, k1=0, k2=0):
        if k1 == 0:
            k1 = self.K1
        if k2 == 0:
            k2 = self.K2

        if not self.NMF_finished:
            print("staNMF Error: runNMF is not complete")
            return

        numPatterns = np.arange(k1, k2+1)
            
        for k in numPatterns:
            path = os.path.join("./staNMFDicts", str(self.folderID), f"K={k}")
                
            Dhat = [
                pd.read_csv(
                    os.path.join(path, f"factorization_{rep}.csv"), 
                    header=None
                ).drop(0, axis=1).values
                for rep in self.replicates
            ]

            distMat = np.zeros((len(self.replicates), len(self.replicates)))
            for i in range(len(self.replicates)):
                distMat[i, i:] = self._instability_worker((i, Dhat, k))

            self.instabilitydict[k] = np.sum(distMat) / (len(self.replicates) * (len(self.replicates)-1))

            if self.parallel:
                with open(os.path.join(path, "instability.csv"), "w") as outputfile:
                    outputwriter = csv.writer(outputfile)
                    outputwriter.writerow([k, self.instabilitydict[k]])
            else:
                with open("instability.csv", "w") as outputfile:
                    outputwriter = csv.writer(outputfile)
                    for i in sorted(self.instabilitydict):
                        outputwriter.writerow([i, self.instabilitydict[i]])

    def get_instability(self):
        if self.stability_finished:
            return self.instabilitydict
        else:
            print("Instability has not yet been calculated for your NMF results. Use staNMF.instability() to continue.")
    
    def ClearDirectory(self, k_list):
        '''
        A storage-saving option that clears the entire directory of each K
        requested, including the instability.csv file in each folder.

        Parameters
        ----------
        k_list : list
            list of K's to delete corresponding directories of

        Notes
        -----
        This should only be used after stability has been calculated for
        each K you wish to delete.
        '''
        for K in k_list:
            path = os.path.join("./staNMFDicts", str(self.folderID), f"K={K}")
            shutil.rmtree(path)

    def plot(self, dataset_title="Dataset Instability Plot", xmax=0,
             xmin=-1, ymin=0, ymax=0, xlab="K", ylab="Instability Index"):
        '''
        Plots instability results for all K's between and including K1 and K2
        with K on the X axis and instability on the Y axis

        Arguments:

        :param: dataset_title (str, optional, default "Dataset Instability Plot")

        :param: ymax (float, optional,  default
        largest Y + (largest Y/ # of points)

        :param: xmax (float, optional, default K2+1)

        :param: xlab (string, default "K") x-axis label

        :param: ylab (string, default "Instability Index") y-axis label

        Returns: None, saves plot as <dataset_title>.png

        Usage: Called by user to generate plot
        '''
        kArray = []

        if self.parallel:
            for K in range(self.K1, self.K2):
                kpath = os.path.join("./staNMFDicts", str(self.folderID), f"K={K}", "instability.csv")
                df = pd.read_csv(kpath)
                kArray.append(int(df.columns[0]))
                self.instabilityarray.append(float(df.columns[1]))
        else:
            for i in sorted(self.instabilitydict):
                kArray.append(i)
                self.instabilityarray.append(self.instabilitydict[i])
        
        if xmax == 0:
            xmax = self.K2 + 1
        if xmin == -1:
            xmin = self.K1
        ymin = 0
        ymax = max(self.instabilityarray) + (max(self.instabilityarray) /
                                             len(self.instabilityarray))
        plt.plot(kArray, self.instabilityarray)
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(str('Stability NMF Results: Principal Patterns vs.'
                      'Instability in ' + dataset_title))
        plotname = str(dataset_title + ".png")
        plt.savefig(plotname)
        plt.show()