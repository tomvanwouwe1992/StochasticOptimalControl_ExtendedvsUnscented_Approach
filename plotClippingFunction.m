% Dynamics of stochastic problem ( xdot = f(x,u,...) )
clear all; close all;
clc;

sharpnessFactor = 50;
lowerClippingValue = 0;
upperClippingValue = 250;

F_control_1 = -250:0.01:400;
F_control_1_clipped = smoothClipping_tanh(F_control_1,sharpnessFactor,lowerClippingValue,upperClippingValue);

figure(1)
subplot(1,2,1)
plot(F_control_1,F_control_1_clipped,'k','LineWidth',2)
xlim([-100 350])
ylim([-25 275])
ylabel('f_{clip}(T^+)')
xlabel('T^+')
box off

subplot(1,2,2)
plot(F_control_1,F_control_1_clipped,'k','LineWidth',2)
xlim([-0.1 0.1])
ylim([-0.1 0.1])
ylabel('f_{clip}(T^+)')
xlabel('T^+')
box off