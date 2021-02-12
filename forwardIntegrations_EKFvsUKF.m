clear all; close all;

noiseVariances = [1 2 5 10 20 50 100 200 500 1000 2000 5000 ];


for i = 1:length(noiseVariances)
    load(['EKFClipped_Variance' num2str(noiseVariances(i)) '.mat']);
    result = forwardIntegration(pendulumResult_UKF);
    save(['Forward_EKF_Variance' num2str(noiseVariances(i)) '.mat'],'result');
end


for i = 1:length(noiseVariances)
    load(['UKFClipped_Variance' num2str(noiseVariances(i)) '.mat']);
    result = forwardIntegration(pendulumResult_UKF);
    save(['Forward_UKF_Variance' num2str(noiseVariances(i)) '.mat'],'result');
end