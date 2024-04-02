num_val_images = 1153;
Q = zeros(num_val_images,4);

for i = 1:num_val_images
    fused=imread(['FusionINN/Fused/fused_' num2str(i-1) '.png']);
    input1=imread(['FusionINN/T1ce/T1ce_' num2str(i-1) '.png']);
    input2=imread(['FusionINN/Flair/Flair_' num2str(i-1) '.png']);
    Q(i,:) = fusionAssess(input1,input2,fused);
end

save('FusionINN/Q.mat', 'Q');

