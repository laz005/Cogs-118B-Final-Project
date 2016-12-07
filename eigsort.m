% [Vsort,Dsort] = eigsort(V,D)
%
% Sorts a matrix eigenvectors and a matrix of eigenvalues in order 
% of eigenvalue size, largest eigenvalue first and smallest eigenvalue
% last.
%
% Example usage:
% [V,D] = eig(A);
% [Vnew,Dnew] = eigsort(V,D);
%
% Edited Jacob Olson 2016

function [Vsort,Dsort] = eigsort(V,D)
eigvals = diag(D);

% Sort the eigenvalues from largest to smallest. Store the sorted
% eigenvalues in the column vector lambda.
[lambda,index] = sort(eigvals,'descend');
Dsort = diag(lambda);

% Sort eigenvectors to correspond to the ordered eigenvalues. Store sorted
% eigenvectors as columns of the matrix vsort.
Vsort = V(:,index);


