function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m,1) X];          %Column of 1's for the bias units for the input layer

Z_2 =  Theta1*X';      %column(i) contains the 'z' values for the i'th training example 

A_2 = sigmoid(Z_2);   %column(i) conatains the activation values for the i'th training example

A_2 = [ones(1,m) ; A_2];            %Bias units for the first hidden layers

Z_3 =  Theta2*A_2;            %column(i) contains the 'z' values for the i'th training example

A_3 = sigmoid(Z_3);    %Column(i) contains the classifier outputs for the i'th training example   

[max_values indices] = max(A_3);

p = indices';

% =========================================================================


end
