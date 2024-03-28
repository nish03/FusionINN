num_val_images = 1153;
Q = zeros(num_val_images,5);

for i = 1:num_val_images
    fused=imread(['DDFM/output/recon/' num2str(i-1) '.png']);
    input1=imread(['INN/T1ce/T1ce_' num2str(i-1) '.png']);
    input2=imread(['INN/Flair/Flair_' num2str(i-1) '.png']);
    Q(i,:) = fusionAssess(input1,input2,fused);
    disp(i);
end

save('/home/h1/s8993054/INN_Fusion/DDFM/Q.mat', 'Q');

