function scd_value = SCD(image_F, image_A, image_B)
    image_F = double(image_F);
    image_A = double(image_A);
    image_B = double(image_B);
    
    % Calculate the differences
    imgF_A = image_F - image_A;
    imgF_B = image_F - image_B;

    % Calculate correlations
    corr1 = sum((image_A(:) - mean(image_A(:))) .* (imgF_B(:) - mean(imgF_B(:)))) / ...
            sqrt(sum((image_A(:) - mean(image_A(:))).^2) * sum((imgF_B(:) - mean(imgF_B(:))).^2));
      
    corr2 = sum((image_B(:) - mean(image_B(:))) .* (imgF_A(:) - mean(imgF_A(:)))) / ...
            sqrt(sum((image_B(:) - mean(image_B(:))).^2) * sum((imgF_A(:) - mean(imgF_A(:))).^2));
      
    % Calculate the sum of correlations of differences
    scd_value = corr1 + corr2;
end