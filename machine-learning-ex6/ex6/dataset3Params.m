function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_mat = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_mat =[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

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
minError = Inf;
curC = Inf;
cur_sigma = Inf;
for i =1:size(C_mat)
for j =1:size(sigma_mat)
model=svmTrain(X, y, C_mat(i),@(x1,x2)gaussianKernel(x1,x2,sigma_mat(j)));
predictions = svmPredict(model, Xval);
error=mean(double(predictions ~= yval));
if error < minError
			minError = error;
			curC = C_mat(i);
			cur_sigma = sigma_mat(j);
		end
endfor
endfor
C=curC;
sigma=cur_sigma;





% =========================================================================

end
