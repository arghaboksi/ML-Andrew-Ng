function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2); % number of features
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

predictions = sigmoid(X*theta);
cost_vector = -(y .* log(predictions) + (1 - y) .* log(1 - predictions));
regularization_term =  sum(theta(2:n,  :).^2) * (lambda/(2*m));
J = (sum(cost_vector) / m ) + regularization_term;

grad_without_regularization =  ((predictions - y)' * X)' ./ m;
regularization_part =  [0; (lambda/m)*theta(2:n, :)];
grad =  grad_without_regularization + regularization_part;

% =============================================================

end
