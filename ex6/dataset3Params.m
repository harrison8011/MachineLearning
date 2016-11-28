function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% initialize the test set for C and sigma, the result matrix

% set_c = [0.01 0.03 0.1 0.3 1 3 10 30];
% set_sigma = [0.01 0.03 0.1 0.3 1 3 10 30];
% result_matrix = eye (size(set_c) * size(set_sigma), 3);
result_matrix = eye (64, 3);
row_num = 0;

% loop both of the two sets to get the prediction error for each group
% save the results into the result matrix
for set_c = [0.01 0.03 0.1 0.3 1 3 10 30]
    for set_sigma = [0.01 0.03 0.1 0.3 1 3 10 30]

        row_num++ ;

        model= svmTrain(X, y, set_c, @(x1, x2) gaussianKernel(x1, x2, set_sigma));
        predictions = svmPredict (model, Xval) ;
        error_pred = mean (double (predictions ~= yval)) ;

        result_matrix (row_num, :) = [ set_c, set_sigma, error_pred];

    end
end

% sort the result matrix by the predictions error
sort_result = sortrows (result_matrix, 3);

C = sort_result (1, 1);
sigma = sort_result (1, 2);

% =========================================================================

end
