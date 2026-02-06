%{
                    EECE5644 FALL 2025 - ASSIGNMENT 4
                                QUESTION 2
                        GMM-Based Image Segmentation
%}
clear all, close all,

fprintf('ASSIGNMENT 4 - QUESTION 2\n');
fprintf('GMM-Based Image Segmentation\n\n');

% Set random seed for reproducibility
rng(5644, 'twister');

% LOAD IMAGE
image_filename = '241004.jpg';  
img_data = imread(image_filename);

% Get image dimensions
fprintf('Image dimensions:')
[n_rows, n_cols, n_channels] = size(img_data);
fprintf('Image size: %d x %d pixels (color)\n', n_rows, n_cols);
fprintf('Channels: %d (RGB)\n', n_channels);

N = n_rows * n_cols;  % Total number of pixels
fprintf('Total pixels: %d\n\n', N);

% FEATURE EXTRACTION
fprintf('FEATURE EXTRACTION\n');
img_data = double(img_data);

% Create row and column index matrices - Adapted from Professor's ExampleKmeansImageSegmentation.m
row_indices = [1:n_rows]' * ones(1, n_cols);
col_indices = ones(n_rows, 1) * [1:n_cols];

% Extract features: [row, col, R, G, B] - Adapted from Professor's ExampleKmeansImageSegmentation.m
features = [row_indices(:)'; col_indices(:)']; 
for channel = 1:n_channels
    channel_data = img_data(:,:,channel);
    features = [features; channel_data(:)'];
end

fprintf('Feature vector dimension: %d (row, col, R, G, B)\n', size(features, 1));

% Normalize each feature to [0,1] - Adapted from Professor's ExampleKmeansImageSegmentation.m
min_features = min(features, [], 2);
max_features = max(features, [], 2);
ranges = max_features - min_features;
x = diag(ranges.^(-1)) * (features - repmat(min_features, 1, N));
fprintf('Features normalized to [0,1] hypercube\n');
fprintf('Data matrix: %d features x %d pixels\n\n', size(x, 1), size(x, 2));

% GMM MODEL ORDER SELECTION
fprintf('GMM MODEL ORDER SELECTION\n');
% Cross-validation parameters
K = 10;  
M_candidates = 2:10;  % Test GMM orders from 2 to 10 components

fprintf('K-fold CV: K = %d\n', K);
fprintf('Candidate model orders: ');
fprintf('%d ', M_candidates);
fprintf('\n\n');

% K-fold cross-validation
fprintf('%d-fold cross-validation:\n', K);

% Create random permutation for CV partitioning - Added based on suggestion from Chatgpt
perm_indices = randperm(N);
x_shuffled = x(:, perm_indices);

% Partition data into K folds - Adapted from Professor's PolynomialFitCrossValidation.m
dummy = ceil(linspace(0, N, K+1));
for k = 1:K
    ind_partition_limits(k,:) = [dummy(k)+1, dummy(k+1)];
end

% Storage for cross-validation results
avg_log_likelihood = zeros(length(M_candidates), 1);

for m_idx = 1:length(M_candidates)
    M = M_candidates(m_idx);
    fprintf('For M = %d components, ',M);
    
    fold_log_likelihoods = zeros(K, 1);
    
    for k = 1:K
        % Partition data - Adapted from Professor's PolynomialFitCrossValidation.m
        ind_validate = ind_partition_limits(k,1):ind_partition_limits(k,2);
        x_validate = x_shuffled(:, ind_validate);
        
        if k == 1
            ind_train_fold = ind_partition_limits(k,2)+1:N;
        elseif k == K
            ind_train_fold = 1:ind_partition_limits(k,1)-1;
        else
            ind_train_fold = [1:ind_partition_limits(k,1)-1, ind_partition_limits(k,2)+1:N];
        end
        
        x_train_fold = x_shuffled(:, ind_train_fold);
        
        % Fit GMM using EM algorithm - Adapted from discussion with Claude 
        % Using MATLAB's fitgmdist with appropriate options
        options = statset('MaxIter', 500, 'Display', 'off');
        try
            gmm_model = fitgmdist(x_train_fold', M, ...
                'RegularizationValue', 1e-5, ...
                'Options', options, ...
                'Start', 'plus');  % k-means++ initialization
            % Compute validation log-likelihood
            fold_log_likelihoods(k) = sum(log(pdf(gmm_model, x_validate') + 1e-10));
        catch
            % If fitting fails, assign very low likelihood
            fold_log_likelihoods(k) = -inf;
            fprintf('(fit failed) ');
        end
    end
    
    % Average validation log-likelihood across folds
    avg_log_likelihood(m_idx) = mean(fold_log_likelihoods);
    fprintf('avg log-likelihood = %.2f\n', avg_log_likelihood(m_idx));
end

% Select optimal model order
[best_log_likelihood, best_idx] = max(avg_log_likelihood);
M_optimal = M_candidates(best_idx);

fprintf('Optimal model order: M* = %d components\n', M_optimal);
fprintf('Best avg log-likelihood: %.2f\n\n', best_log_likelihood);

% Visualize cross-validation results
figure(1), 
plot(M_candidates, avg_log_likelihood, 'b-o', 'LineWidth', 2, 'MarkerSize', 6); hold on,
plot(M_optimal, best_log_likelihood, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Number of GMM Components (M)');
ylabel('Average Validation Log-Likelihood');
title('Model Order Selection via 10-Fold CV');
legend('CV Log-Likelihood', sprintf('M* = %d', M_optimal), 'Location', 'best');
grid on;

% TRAIN FINAL GMM WITH OPTIMAL MODEL ORDER
fprintf('\nTRAINING FINAL GMM\n');

% Train final model on all data - Adapted from discussion with Claude 
options_final = statset('MaxIter', 1000, 'Display', 'off');

gmm_final = fitgmdist(x', M_optimal, ...
    'RegularizationValue', 1e-5, ...
    'Options', options_final, ...
    'Start', 'plus');

fprintf('Final model:\n');
fprintf('  Components: %d\n', M_optimal);
fprintf('  Mixing proportions: ');
fprintf('%.3f ', gmm_final.ComponentProportion);
fprintf('\n\n');

% PIXEL ASSIGNMENT AND SEGMENTATION - Adapted from discussion with Claude
fprintf('IMAGE SEGMENTATION\n');

% Compute posterior probabilities for each pixel
posteriors = posterior(gmm_final, x');

% Assign each pixel to most likely component - Adapted from Professor's ExampleKmeansImageSegmentation.m
[~, pixel_labels] = max(posteriors, [], 2);

% Reshape labels to image dimensions - Adapted from Professor's ExampleKmeansImageSegmentation.m
label_image = reshape(pixel_labels, n_rows, n_cols);

fprintf('Pixels per segment:\n');
for m = 1:M_optimal
    n_pixels_segment = sum(pixel_labels == m);
    fprintf('  Segment %d: %d pixels (%.1f%%)\n', m, n_pixels_segment, ...
        100 * n_pixels_segment / N);
end
fprintf('\n');

% VISUALIZATION - Pattern from Professor's ExampleKmeansImageSegmentation.m
figure(2), clf,

% Original image
subplot(1,2,1);
imshow(uint8(img_data));
title('Original Image', 'FontSize', 14);

% Segmented image with uniformly distributed grayscale values
subplot(1,2,2);
label_image_display = uint8(label_image * 255 / M_optimal);
imshow(label_image_display);
title(sprintf('GMM Segmentation (M*=%d)', M_optimal), 'FontSize', 14);

sgtitle(sprintf('GMM-Based Image Segmentation: %s', image_filename), ...
    'FontSize', 16, 'FontWeight', 'bold');