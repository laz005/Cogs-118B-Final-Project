
%% Load data, find the number of datapoints (N) and attributes (P).
[P,N] = size(cleveland');
degree = goals';

%% 1. Find the sample mean. Zero mean the data.
sampleMean = mean(cleveland);
Z = cleveland'-repmat(sampleMean',[1,N]);

%% 2. Using the zero-meaned data, find the sample covariance matrix.
CV = 1/N*(Z*Z');

%% 3. Find the eigenvectors and eigenvalues of the sample covariance matrix.
[Vp,Dp] = eig(CV);

%% 4. Use the eigsort function to sort the eigenvectors and eigenvalues 
% in order of largest eigenvalue to smallest eigenvalue.
[V, D] = eigsort(Vp,Dp);

% Transform to PC space
C = V'*Z;
covarianceOfPCA = 1/N*(C*C');
disp('Covariance Matrix of PCA is: ');
disp(covarianceOfPCA); % Only high variance in top 3 PCs - plot this.
figure(2)
hold off;
imagesc(covarianceOfPCA);
colorbar;
title('Covariance Matrix of PCA');

% C_hat = C(1:k,:); %(k?p, p = original data dimension) 
C_hat = C(1:5,:); % k = 3 here.

 %Z_hat = V(:,1:k)*C(1:k,:)+repmat(mean(X,2),1,N);%%5=k
Z_hat = V(:,1:3)*C(1:3,:)+repmat(sampleMean',1,N); 

figure(3);
hold on;
for i=1:303
    color = 'k.';
    if(degree(i) == 1)
        color = 'b.';
    elseif(degree(i) == 2)
        color = 'g.';
    elseif(degree(i) == 3)
        color = 'y.';
    elseif(degree(i) == 4)
        color = 'r.';
    end
    scatter3(C_hat(1,i),C_hat(2,i),C_hat(3,i), color);
end
%scatter3(C_hat(1,:),C_hat(2,:),C_hat(3,:), 'k.');
title('Our data reduced to the top 3 principal components and plotted in PC space.');
figure(4);
hold off;
scatter3(Z_hat(1,:),Z_hat(2,:),Z_hat(3,:),'k.');
title('Our data reduced to the top 3 principal components, reconstructed back into feature space.');


