function result = forwardIntegration(pendulumResult)
import casadi.*
import org.opensim.modeling.*

X = MX.sym('X',1,1);
dX = MX.sym('dX',1,1);
F_dist =  MX.sym('F_dist',1,1);
F_1 =  MX.sym('F_1',1,1);
F_2 =  MX.sym('F_2',1,1);

K_1 = MX.sym('K_1',1,1);
B_1 = MX.sym('B_1',1,1);
K_2 = MX.sym('K_2',1,1);
B_2 = MX.sym('B_2',1,1);
    
    sharpnessFactor = 50;
    lowerClippingValue = 0;
    upperClippingValue = 250;
        
    % Dynamic System parameters
    g = 9.81; l = 1.0; m = 75;
    
F_control_1 = F_1 + K_1*(X-pi) + B_1*dX;
F_control_1_clipped = smoothClipping_tanh(F_control_1,sharpnessFactor,lowerClippingValue,upperClippingValue);

F_control_2 = F_2 + K_2*(X-pi) + B_2*dX;
F_control_2_clipped = -smoothClipping_tanh(F_control_2,sharpnessFactor,lowerClippingValue,upperClippingValue);

Xdot = [ dX; (-m*g*l*sin(X) + F_control_1_clipped + F_control_2_clipped + F_dist)/(m*l^2)];
fcn_OL = Function('fcn_OL',{dX,X,F_1,F_2,F_dist,K_1,B_1,K_2,B_2},{Xdot});

t0 = 0;
tf = 100;
dt = 0.01;
N = (tf - t0)/dt;

K_1 = pendulumResult.K_1_sol;
B_1 = pendulumResult.B_1_sol;
K_2 = pendulumResult.K_2_sol;
B_2 = pendulumResult.B_2_sol;
F_1 = pendulumResult.F_1_sol;
F_2 = pendulumResult.F_2_sol;

noiseVariance = pendulumResult.noiseVariance; %N²s

noiseVarianceDiscrete = noiseVariance/dt;
noiseStdDiscrete = sqrt(noiseVarianceDiscrete);
cutOff_3STD = 2*noiseStdDiscrete;

X_sim = NaN(N+1,1); 
X_sim(1) = pendulumResult.X_sol(1)+0.0001;
dX_sim = NaN(N+1,1); 
dX_sim(1) = pendulumResult.dX_sol(1);
F_1_sim = NaN(N,1); 
F_2_sim = NaN(N,1); 
for k = 1:N
    Urf = MX.sym('Urf',6);
    Xdot = Urf(1);
    dXdot = Urf(2);
    
    Xnext = Urf(3);
    dXnext = Urf(4);
    
    Xdotnext = Urf(5);
    dXdotnext = Urf(6);
    
    F_dist = normrnd(0,sqrt(noiseVarianceDiscrete),1,1);
    F_dist(F_dist>cutOff_3STD) = cutOff_3STD;
    F_dist(F_dist<-cutOff_3STD) = -cutOff_3STD;

%     F_dist =0;
    F_control_1 = F_1 + K_1*(X_sim(k)-pi) + B_1*dX_sim(k);
    F_control_1_clipped = smoothClipping_tanh(F_control_1,sharpnessFactor,lowerClippingValue,upperClippingValue);
    F_1_sim(k) = F_control_1_clipped;
    
    F_control_2 = F_2 + K_2*(X_sim(k)-pi) + B_2*dX_sim(k);
    F_control_2_clipped = -smoothClipping_tanh(F_control_2,sharpnessFactor,lowerClippingValue,upperClippingValue);
    F_2_sim(k) = F_control_2_clipped;
    
    rf = rootfinder('rf','newton',struct('x',Urf,'g',[fcn_OL(dX_sim(k),X_sim(k),F_1,F_2,F_dist,K_1,B_1,K_2,B_2)-[Xdot;dXdot]; ...
                                                      fcn_OL(dXnext,Xnext,F_1,F_2,F_dist,K_1,B_1,K_2,B_2)-[Xdotnext;dXdotnext]; ... 
                                                      [Xnext;dXnext] - ([X_sim(k);dX_sim(k)] + 0.5*dt*[Xdot;dXdot] + 0.5*dt*[Xdotnext;dXdotnext])]),struct('abstol',1e-16));
    solution_u = rf(zeros(6,1),[]);
    solution_u = full(solution_u);
    X_sim(k+1) = solution_u(3);
    dX_sim(k+1) = solution_u(4);
    
end

result.X_sim = X_sim;
result.dX_sim = dX_sim;
result.F_1_sim = F_1_sim;
result.F_2_sim = F_2_sim;

result.rmsSway = rms((result.X_sim - pi)*180/pi);
result.rmsSwayVel = rms((result.dX_sim)*180/pi);

