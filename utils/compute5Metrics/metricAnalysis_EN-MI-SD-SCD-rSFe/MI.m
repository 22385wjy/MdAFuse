function mi_value = MI(image_F, image_A, image_B)
    
    image_F = double(image_F) / 255;
    image_A = double(image_A) / 255;
    image_B = double(image_B) / 255;

    mi_A = mutual_information(image_F(:), image_A(:));
    mi_B = mutual_information(image_F(:), image_B(:));

    mi_value = mi_A + mi_B;
end

function mi = mutual_information(image1, image2)
    
    num_bins = 256; % 
    joint_hist = histcounts2(image1, image2, num_bins, 'Normalization', 'probability');

    joint_prob = joint_hist / sum(joint_hist(:));

    marginal_hist1 = sum(joint_hist, 2);
    marginal_hist2 = sum(joint_hist, 1);

    marginal_prob1 = marginal_hist1 / sum(marginal_hist1);
    marginal_prob2 = marginal_hist2 / sum(marginal_hist2);

    eps = 1e-10;
    joint_prob(joint_prob < eps) = eps;
    marginal_prob1(marginal_prob1 < eps) = eps;
    marginal_prob2(marginal_prob2 < eps) = eps;

    mi = sum(sum(joint_prob .* log2(joint_prob ./ (marginal_prob1 * marginal_prob2))));
end