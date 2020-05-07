function [net] = biasBackProp(ni,nh,no,eta,sse,x,t)

% Ashikur Rahman

% This program implements back propagation algorithm to train 3 layer (input,hidden,output)neural network.  
% ni = number of input or size of input vector [Positive int]
% nh = number of hidden layer nuron [Positive int]
% no = number of output layer nuron [Positive int]
% x  = input, each column is a feature vector, total number of column indicates
% number of patterns ,total row number indicate dimension of the feature vector
% or input.
% eta = learning rate, typical value I used (.05 ~ 2), but it depends
% sse = sum squared error threshold, training is finished if sse < sum squared error
% t = desired output.
% net = structure of network results, weights and some other data to simulate after
% training
% The program uses sigmoid activation function.
% Network: wih -> input to hidden layer weight
% who -> hidden to output layer weight')

%-------------------------net---------------------------------%


% net.wih = wih;
% net.who = who;
% net.ni = ni;
% net.nh = nh;
% net.no = no;


% ---------------------------Example-----------------------------%

% Train the network for function : (A+B)C

% We denote 

% x(1,:) = A
% x(2,:) = B
% x(3,:) = C
% 
% Create input patterns 
% 
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



clear net

if ni ~= length(x(:,1))
 error('Input dimension does not match with number of input "ni"')
end

if no ~= length(t(:,1))
 error('Output dimension: "t" does not match with number of output "no"')
end

if no <= 0
 error('This program needs at least one hidden layer neuron, please input  "nh" 1 or higher')
end

if no <= 0 || nh<=0 || ni<=0
 error('Please input positive integer number as your input, hidden and output neuron numbers')
end


% t=t/max(t)
% for i = 1:length(x(1,:))
% if sum(x(:,i))~=0
%     x(:,i)=x(:,i)/sum(x(:,i))
% end
% end

% [I N] = size(x)
% minp = repmat(min(x,[],2),1,N);
% maxp = repmat(max(x,[],2),1,N);
% 
% x = 0 +(1-0).*(x-minp)./(maxp-minp)
% 
% 
% 
% [I N] = size(t)
% minp = repmat(min(t,[],2),1,N);
% maxp = repmat(max(t,[],2),1,N);
% 
% t = 0 +(1-0).*(t-minp)./(maxp-minp)
% 

X1n = log10(1+x(:,1))./log10(1001);
x(:,1)= X1n;
tn = log10(1+t)/log10(1001);
t = tn;



 c=0; % Iteration counter
 wih = 0.1*randn(nh,ni+1); % initialize weight vectors and bias weights from input to hidden layer
 who = 0.1*randn(no,nh+1); % initialize weight vectors and bias weights from hidden to output layer
%  err = 0; % Error SSE
%  
 netj = zeros(1,nh); %input to hidden layer activation
 outj = zeros(1,nh); %output of hidden layer activation
 netk = zeros(1,no); %input to output layer activation
 outk = zeros(1,no); %output of hidden layer activation
 delk = zeros(1,no); % local gradient for output layer neuron
 delj = zeros(1,nh); % local gradient for hidden layer neuron
 
 lambda = 1;
 
 while(c<1000000) % loop until 10000 iteration or SSE has reached to limit to break it.
     c=c+1;
     err = 0;
     for i = 1:length(x(1,:)) %loop for each pattern

% hidden layer output 
         for j = 1:nh
            netj(j) = wih(j,1:end-1)*double(x(:,i))+wih(j,end)*1; %input to hidden layer neurons
             outj(j) = 1./(1+exp(-1*netj(j))); % output of hidden layer nerons
%             outj(j) = 2/(1+exp(-2*netj(j)))-1;
%           
%             outj(j)=lambda*(abs(netj(j))*1/(1+exp(-1*(netj(j)))));


        end

% hidden to output layer

        for k = 1:no
            
            netk(k) = who(k,1:end-1)*outj'+who(k,end)*1;
              outk(k) = 1./(1+exp(-1*netk(k))); % output
%              outk(k)=lambda*(abs(netk(k))*1/(1+exp(-1*(netk(k)))));

%             delk(k) = lambda*abs(outk(k))*outk(k)*(1-outk(k))*(t(k,i)-outk(k));


% outk(k) = netk(k);
% delk(k) = (t(k,i)-outk(k));
 delk(k) = outk(k)*(1-outk(k))*(t(k,i)-outk(k));
%             lambda*(abs(y)*y*(1-y))
            err = err+.5*(t(k,i)-outk(k))^2; 

        end
        
        
        

        
        % back proagation
        for j = 1:nh
            s=0;
            for k = 1:no
                s = s+who(k,j)*delk(k);
            end
    
%             delj(j) = lambda*abs(outj(j))*outj(j)*(1-outj(j))*s; 
  delj(j) = outj(j)*(1-outj(j))*s;
            s=0;
        end

 
                % update weight hidden-output


        for k = 1:no
            for l = 1:nh
                who(k,l)=who(k,l)+eta*delk(k)*outj(l)/(1-.1); % Momentum & learning rate included
            end
            who(k,l+1)=who(k,l+1)+eta*delk(k)*1/(1-.1);
        end

        

        
        
% update weight input-hidden

        


for j = 1:nh
            for ii = 1:ni
                wih(j,ii)=wih(j,ii)+eta*delj(j)*double(x(ii,i))/(1-.1);
            end
            wih(j,ii+1)=wih(j,ii+1)+eta*delj(j)*1/(1-.1);

        end



     end
    if err<sse, break , end

 end
 
 % If loop does not converge it will run 10000 whole iterations. Convergence
 % within 10000 iterations depends upon lot of factors, learning rate,sse,
 % pattern type and other factors
 
 
 disp('number of iteration processed: ')
 c
 
 disp('number of input, hidden and output layer nuron: ')
 ni
 nh
 no
 
 disp('Sum squared error input: sse')
 sse
 
 disp('Sum squared error in convergence: err')
 err
 
 
 
 disp('learning rate: eta')
 eta
 


net.wih = wih;
net.who = who;
net.ni = ni;
net.nh = nh;
net.no = no;

 disp('network: wih -> input to hidden layer weight, who -> hidden to output layer weight')
 disp('last column in wih and who is bias weights')
 disp('Now try to simulate with bpSim: bpSim(your_input,net)')
 
net

% Error XOR

% [bpSim([0;0],net) bpSim([0;1],net) bpSim([1;0],net) bpSim([1;1],net)]



