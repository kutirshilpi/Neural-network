
function [out] = bpSim(x,net)

% Ashikur Rahman
% This program simulate a trained network using the 'net', the output of 'biasBackProp'
% Network: wih -> input to hidden layer weight
% who -> hidden to output layer weight')
% nh = number of hidden layer nuron [Positive int]
% no = number of output layer nuron [Positive int]
% outk is output

% ---------------------------Example-----------------------------%

% Train the network for function : (A+B)C

% We denote 

% x(1,:) = A
% x(2,:) = B
% x(3,:) = C
% 
% Create input patterns 

% x =  [0     0     0     0     1     1     1     1
%       0     0     1     1     0     0     1     1
%       0     1     0     1     0     1     0     1]

% % % 
% Create or generate or provide desired output
% t=and(or(x(1,:),x(2,:)),x(3,:))

% Desired output

% t =
% 
%      0     0     0     1     0     1     0     1
% 

% This program does the training work, after training result is returened as net.
% The program bpSim.m can simulate any single pattern using the results of this program.
% 
% ............. Command Line Simulation Example...............................%

% >> net = biasBackProp(3,3,1,1,0.001,x,t);

% >> bpSim([0;1;1],net)
% 
% ans =
% 
%     0.9758

wih = net.wih;
who = net.who;
nh = net.nh;
no = net.no;


for j = 1:nh
            netj(j) = wih(j,1:end-1)*double(x)+wih(j,end)*1;
             outj(j) = 1./(1+exp(-1*netj(j)));
%             outj(j) = 2/(1+exp(-2*netj(j)))-1;
%             outj(j)=lambda*(abs(netj(j))*1/(1+exp(-1*(netj(j)))));

end

% hidden output layer

for k = 1:no
            
     netk(k) = who(k,1:end-1)*outj'+who(k,end)*1;
%      outk(k) = netk(k);
       outk(k) = 1./(1+exp(-1*netk(k)));
%      outk(k)=lambda*(abs(netk(k))*1/(1+exp(-1*(netk(k)))));

outk(k) = 10^(outk(k)*log10(1001))-1;

     disp('Actual output ---- Rounded Output')
     out = [outk(k) round(outk(k))];

end

