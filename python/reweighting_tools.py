from hep_ml.reweight import GBReweighter
import numpy as np

class BDTReweighter(GBReweighter):

    def __init__(self,
                 n_estimators=40,
                 learning_rate=0.2,
                 max_depth=3,
                 min_samples_leaf=200,
                 loss_regularization=5.,
                 gb_args=None,
                 verbose=True):
        super().__init__(
                 n_estimators,
                 learning_rate,
                 max_depth,
                 min_samples_leaf,
                 loss_regularization,
                 gb_args)
        self.normalization = 1.
        self.verbose = verbose

    def CombineAndShuffle(self, samples, weights):
        X_combined = np.concatenate(samples, axis=0)
        w_combined = np.concatenate(weights, axis=0)
        indices = np.arange(len(X_combined))
        np.random.shuffle(indices)
        X_combined = X_combined[indices]
        w_combined = w_combined[indices]    

        return X_combined, w_combined

    def predict_weights(self, original):
        # modify predict weights so that it also applies the correct normalization to the weights
        re_weights = super().predict_weights(original)
        re_weights *= self.normalization

        return re_weights

    def fit(self, original, target, original_weight=None, target_weight=None):

        # default method does not give correct results for cases where original weights have negative values
        # If original does not contain any negative weights then just use the default method
        if original_weight is None or not np.any(original_weight < 0):
            if self.verbose: print('BDTReweighter: No negative weights present for original, fitting model')
            super().fit(original, target, original_weight, target_weight)
            if self.verbose: print('BDTReweighter: Finished fitting model')
            normalization_num = target_weight.sum() if target_weight is not None else target.sum()
            #normalization_denom = original_weight.sum() if original_weight is not None else original.sum()
            normalization_denom = super().predict_weights(original).sum()
            self.normalization = normalization_num/normalization_denom 

        # If original does have negative weights then we apply the work-around
        else:     
            if self.verbose: print('BDTReweighter: Negative weights present in original')
            if self.verbose: print('BDTReweighter: Performing first fit to reweight original to remove the negative weights')
            # split original into events with positive and negative weights
            positive_mask = original_weight >= 0
            negative_mask = original_weight < 0

            original_pos_weight = original_weight[positive_mask]
            original_neg_weight = original_weight[negative_mask]

            original_pos = original[positive_mask]
            original_neg = original[negative_mask]

            # we derive weights to reweight the positive events to positive-negative 
            negwt_reweighter = GBReweighter(n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                   loss_regularization=self.loss_regularization,gb_args=self.gb_args)
            # target in this case is positive events - negative events 
            step1_target, step1_target_weight = self.CombineAndShuffle((original_pos, original_neg), (original_pos_weight, original_neg_weight)) 
            negwt_reweighter.fit(original_pos, step1_target, original_pos_weight, step1_target_weight)
            re_weight_step1 = negwt_reweighter.predict_weights(original_pos)*original_pos_weight
            re_weight_step1 *= step1_target_weight.sum()/re_weight_step1.sum()

            if self.verbose: print('BDTReweighter: Finished first fit to reweight original to remove the negative weights')


            if self.verbose: print('BDTReweighter: Performing second fit to reweight to the target')
            # now we train a model to reweight the reweighted positive original to the target

            super().fit(original_pos, target, re_weight_step1, target_weight)
            if target_weight is None: target.sum()/(super().predict_weights(original_pos)*re_weight_step1).sum()
            else: self.normalization = target_weight.sum()/(super().predict_weights(original_pos)*re_weight_step1).sum()
            if self.verbose: print('BDTReweighter: Finished performing second fit to reweight to the target')
