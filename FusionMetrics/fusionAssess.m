function res=fusionAssess(im1,im2,fused)

% function res=fusionAssess(im1,im2,fused)
%
% This function is to assess the fused image with different fusion
% assessment metrics.
% 
% im1   ---- input image one;
% im2   ---- input image two;
% fused ---- the fused image(s)
% res   ==== the metric value
%
% Z. Liu @ NRCC [Aug 21, 2009]
%

im1=double(im1);
im2=double(im2);
fused=double(fused);

% calculate the image fusion metrics:


% feature mutual informtion $Q_{FMI}$
Q(1)=fmi(im1,im2,fused);
    
% Wang - NCIE $Q_{NCIE}$
Q(2)=metricWang(im1,im2,fused);
    
% Xydeas $Q_{XY}$
Q(3)=metricXydeas(im1,im2,fused);
    
% Piella $Q_P$
Q(4)=metricPeilla(im1,im2,fused,1);
    
res=Q;
