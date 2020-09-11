function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1: Cost Function Value after Forward Propagation
X = [ones(m,1) X];        %Add a column of 1's for the bias unit

Z2 = Theta1 * X';      

A2 = sigmoid(Z2);     %Each column of A2 is the activations of the hidden units 
                      %in the first hidden layer w.r.t each training examples
                      
A2 = [ones(1,m); A2]; %Add a row of 1's for the bias unit

Z3 = Theta2 * A2;

A3 = sigmoid(Z3);    %Column(i) contains the output for ith training example

%For Implementation purpose each y(i) should be a vector
%column(i) of yVector is the actual output,y(i) transformed into a vector of 1's and 0's
yVector = zeros(num_labels,m);
for i = 1:m
  yVector_i = (1:num_labels)';
  yVector(:,i) = yVector_i;
endfor

yVector = (yVector == y');

%Calculation of associated cost for each training example
costForEachExample = sum(-((log(A3) .* yVector) + ((1-yVector) .* log(1-A3))));   %column(i) contains the cost for ith training example
totalCost = sum(costForEachExample);
J = totalCost/m;

%Cost function value with regularization

% Calculation for Theta One terms
ThetaOneSquares = Theta1(:,2:input_layer_size+1) .^ 2;
sumOfSquaresThetaOne = sum(sum(ThetaOneSquares));

%Calculation for Theta Two terms
ThetaTwoSquares = Theta2(:,2:hidden_layer_size+1) .^ 2;
sumOfSquaresThetaTwo = sum(sum(ThetaTwoSquares));

regularizationTerm = ((sumOfSquaresThetaOne + sumOfSquaresThetaTwo)*lambda)/(2*m);

J = J + regularizationTerm;



% Part Two : Backpropagation without regularization 
%Already calculated Z2,A2,Z3 and A3 while calculating cost function value, 
%we can use those values for backpropagation

delta3 = A3 - yVector;      %delta terms associated with third layer for all training examples

delta2 = (Theta2'*delta3)(2:hidden_layer_size+1,:) .* sigmoidGradient(Z2);  %delta terms associated with second layer for all training examples
                                                                            %avoiding the delta terms for the bias units
Theta2_grad = (delta3*A2')/m;

Theta1_grad = (delta2*X)/m;

%Part Three : Regularized Neural Networks
%For regularization we do not update the gradients of weights corresponding to the bias terms
Theta1_grad(:,2:input_layer_size+1) = Theta1_grad(:,2:input_layer_size+1) + ((lambda/m)*Theta1(:,2:input_layer_size+1));
Theta2_grad(:,2:hidden_layer_size+1) = Theta2_grad(:,2:hidden_layer_size+1) + ((lambda/m)*Theta2(:,2:hidden_layer_size+1));

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
