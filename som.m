function main

    close all
    %rng(5)
    
    d=input('Choose a data set (1 through 3):  ');
    
    switch d
        case 1
            X=load("data1.txt"); % 255-D, 10 class handwritten number data
            % labels for handwritten digit data, class 1 corresponds to
            % digit 0, class 2 corresponds to digit 1, ...
            y=zeros(2240,1);
            y(1:224)=1;
            y(225:448)=2;
            y(449:672)=3;
            y(673:896)=4;
            y(897:1120)=5;
            y(1121:1344)=6;
            y(1345:1568)=7;
            y(1569:1792)=8;
            y(1793:2016)=9;
            y(2017:2240)=10;
        case 2
            X=SOMColorData(3);
        case 3
            X=load("CityLatLong.txt"); % latitude and longitude of 17 US and Canada Cities in minutes and seconds and decimals
            X=X(:,5:6); % decimal values for latitude and longitude are 5th and 6th cols
            X=[X(:,2) X(:,1)]; % convert to x-y coordinates.  Longitude is x, Latitude is y
    end
    if d==1
        [yp, m]=Kmeans(X);
        CmKmeans=ConfusionMatrix(y,yp)
        [l w K]=size(m);
        figure(1)
        for k=1:K
            subplot(3,4,k)
            showImage(m(:,:,k));
        end
        
        [yp, m]=CompetitiveLearningNetwork(X);
        CmCompetitiveLearningNetwork=ConfusionMatrix(y,yp)
        [l w K]=size(m);
        figure(2)
        for k=1:K
            subplot(3,4,k)
            showImage(m(:,:,k));
        end
    elseif d==2
        [m winningNodes]=SOM(X);
        figure(1)
        showSOM(m)
        figure(2)
        showSOMWinners(m, winningNodes)
    else
        m=SOM2(X);
        figure(1)
        hold on
        scatter(X(:,1),X(:,2))
        plot(m(1,:),m(2,:))
        hold off
        axis equal
    end
end


function [y, m]=Kmeans(X) % K-means clustering method
    [N d]=size(X); % number of data points and dimension
    %K=input('Choose a number of clusters to use:  ');
    K=10;
    y=zeros(N,1); % initialize the vector of classifications for each sample
    tolerance=10^-8;
    error=1;
    mnew=X(randi(N,K,1),:); % randomly initialze the K mean vectors for each cluster
    iter=0;
    while error>tolerance
        iter=iter+1;
        mold=mnew; % set old mean vectors to the previous iteration values
        mnew=zeros(K,d); % initialize new mean vectors
        for n=1:N
            min=10^8; % set min distance high initially so it gets overwritten
            for k=1:K
                if norm(X(n,:)-mold(k,:))<min
                    y(n)=k; % set sample to the class with closest mean vector
                    min=norm(X(n,:)-mold(k,:)); % re-set min distance
                end
            end
        end
        % update mean vectors
        for k=1:K
            Xk=find(y==k); % indicies of samples in the class
            Nk=length(Xk); % calculate number of samples in the class
            sum=zeros(1,d);
            for i=1:Nk
                sum=sum+X(Xk(i),:);
            end
            mnew(k,:)=sum/Nk;
        end
        error=norm(mnew-mold);
    end
    % convert mean vectors back into a 16x16 image
    m=zeros(16,16,K);
    for k=1:K
        m(:,:,k)=reshape(mnew(k,:),[16 16]);
    end
end

function [y, m]=CompetitiveLearningNetwork(X) % Clustering using competitive learning neural network
    [N d]=size(X); % number of data points and dimension
    K=10;; % number of clusters
    n=d; % number of input nodes
    m=K; % number of output nodes
    for i=1:N
        X(i,:)=X(i,:)/norm(X(i,:)); % normalize data vectors
    end
    Wnew=X(randi(N,m,1),:)'; % randomly initialize weight vectors to be random samples from data
    error=inf;
    tolerance=10^-8;
    eta=0.9; % learning rate
    alpha=0.999; % decay factor for learning rate
    iter=0;
    while error>tolerance && eta>0.05
        iter=iter+1;
        Wold=Wnew;
        i=randi(N);
        xtrain=X(i,:)'; % randomly select a training sample
        yp=zeros(1,m); % output
        a=zeros(1,m); % activation of each output node
        for i=1:m
            a(i)=Wold(:,i)'*xtrain; % activation
        end
        winner=find(a==max(a)); % winning node is the weight vector closest to the training sample with the highest activation
        yp(winner)=1; % set winning output node to 1
        Wnew=zeros(n,m); % initialize new weight vectors
        for i=1:m
            if i==winner
                Wnew(:,i)=Wold(:,i)+eta*(xtrain-Wold(:,i)); % winning nodes learns, moves closer to training sample
                Wnew(:,i)=Wnew(:,i)/norm(Wnew(:,i)); % re-normalize new weight vector
            else
                Wnew(:,i)=Wold(:,i); % nodes that do not win do not learn
            end
        end
        eta=eta*alpha; % reduce learning rate
        error=norm(Wnew(:,winner)-Wold(:,winner)); % error is difference between old and new weight vectors
    end
    % classify all of the data
    y=zeros(N,1);
    for i=1:N
        a=zeros(1,m); % activation of each output node
        xtrain=X(i,:)';
        for j=1:m
            a(j)=Wnew(:,j)'*xtrain; % activation
        end
        winner=find(a==max(a)); % winning node is the weight vector closest to the training sample with the highest activation
        y(i)=winner; % set classification to be the node with highest activation (closest weight vector)
    end
    % convert the normalized weight vectors back to values between 0 and 255
    for i=1:m
        maxval=max(Wnew(:,i));
        Wnew(:,i)=Wnew(:,i)*255/maxval;
    end
    % convert weight vectors back into a 16x16 image
    m=zeros(16,16,K);
    for k=1:K
        m(:,:,k)=reshape(Wnew(:,k)',[16 16]);
    end
end

% For the parameters used, the Kmeans algorithms seems slightly better than
% the competitive learning network for classifying the handwritten digits

function [m winningNodes]=SOM(X) % Self-Organizing Map (2-D grid)
    [N d]=size(X); % number of data points and dimension
    n=d; % number of input nodes
    SOMsize=30; % width and height of SOM
    m=SOMsize^2; % number of output nodes
    Wnew=randi(255,n,m); % randomly initialize weight vectors
    for i=1:m
        Wnew(:,i)=Wnew(:,i)/norm(Wnew(:,i)); % normalize weight vectors
    end
    winningNodes=[]; % initialize set of winning nodes
    error=inf;
    tolerance=10^-8;
    eta=0.9; % learning rate
    sigma=SOMsize/5; % width parameter for gaussian function
    alpha=0.99; % decay factor for learning rate
    iter=0;
    while error>tolerance && eta>0.1 && sigma>1
        iter=iter+1;
        Wold=Wnew;
        i=randi(N);
        xtrain=X(i,:)'; % randomly select a training sample
        xtrain=xtrain/norm(xtrain); % normalize training sample
        a=zeros(SOMsize,SOMsize); % activation of each output node
        for i=1:SOMsize
            for j=1:SOMsize
                a(i,j)=Wold(:,(SOMsize*(i-1)+j))'*xtrain; % activation
            end
        end
        [row col]=find(a==max(max(a))); % winning node is the weight vector closest to the training sample with the highest activation
        index=randi(length(row)); % in case more than one winning node, pick random node
        row=row(index);
        col=col(index);
        winningNodes=[winningNodes;[row col]]; % add winner to list of winners
        Wnew=zeros(n,m); % initialize new weight vectors
        for i=1:SOMsize
            for j=1:SOMsize
                d=norm([i j]-[row col]); % euclidian distance between winning node and current node in the 2-D array
                u=exp(-d^2/(2*sigma^2)); % weight function for learning depends on distance from winning node
                Wnew(:,(SOMsize*(i-1)+j))=Wold(:,(SOMsize*(i-1)+j))+u*eta*(xtrain-Wold(:,(SOMsize*(i-1)+j)));
                Wnew(:,(SOMsize*(i-1)+j))=Wnew(:,(SOMsize*(i-1)+j))/(norm(Wnew(:,(SOMsize*(i-1)+j)))); % re-normalize new weight vector
            end
        end
        eta=eta*alpha; % reduce learning rate
        sigma=sigma*alpha; % reduce width parameter
        error=norm(Wnew(:,(SOMsize*(row-1)+col))-Wold(:,(SOMsize*(row-1)+col))); % error is difference between old and new winning weight vectors
    end
    % convert the normalized weight vectors back to values between 0 and 255
    for i=1:m
        Wnew(:,i)=Wnew(:,i)*255;
    end
    % convert weight vectors back into a map
    m=zeros(SOMsize,SOMsize,n); % 3-D matrix of RGB values at each point in map
    for i=1:n
        m(:,:,i)=reshape(Wnew(i,:),[SOMsize SOMsize])';
    end
end

function m=SOM2(X) % Self-Organizing Map (TSM problem, nodes are points in 2-D )
    [N d]=size(X); % number of data points and dimension
    n=d; % number of input nodes
    m=40; % number of output nodes
    center=[mean(X(:,1)),mean(X(:,2))]; % center of circle of network weight vectors
    r=15; % radius of circle of network weight vectors
    % arrange output nodes in a circle
    syms x
    circletop=center(2)+sqrt(r^2-(x-center(1))^2);
    circletop=matlabFunction(circletop);
    circlebottom=center(2)-sqrt(r^2-(x-center(1))^2);
    circlebottom=matlabFunction(circlebottom);
    % the plus 1 in xvals and m/2:-1:2 in yvals is to avoid double counting
    % and make the plot continuous
    xvals=linspace(center(1)-r,center(1)+r,m/2+1);
    yvals=[circletop(xvals) circlebottom(xvals((m/2):-1:2))];
    xvals=[xvals xvals((m/2):-1:2)];
    Wnew=[xvals; yvals];
    error=inf;
    tolerance=10^-8;
    eta=0.9; % learning rate
    sigma=2; % width parameter for gaussian function
    alpha=0.999; % decay factor for learning rate
    iter=0;
    pastCities=[]; % stores past training cities used so same one isn't chosen many times in a row
    while eta>0.05 && sigma>0.05 && error>tolerance
        iter=iter+1;
        
%         hold on
%         scatter(X(:,1),X(:,2))
%         plot(Wnew(1,:),Wnew(2,:))
%         hold off
%         axis equal
%         pause
%         close all
        
        Wold=Wnew;
        if length(pastCities)==17 % re-set if all cities have been used
            pastCities=[];
        end
        i=randi(N);
        while ismember(i,pastCities)==1 % pick a new random city
            i=randi(N);
        end
        pastCities=[pastCities i]; % add new city to list of cities used
        xtrain=X(i,:)'; % randomly select a training sample
        a=zeros(1,m); % activation of each output node
        for i=1:m
            a(i)=norm(Wold(:,i)-xtrain); % distance from each weight vector to the training city
        end
        winner=find(a==min(a)); % winning node is the weight vector closest to the training sample with the highest activation
        index=randi(length(winner)); % in case more than one winning node, pick random node
        winner=winner(index);
        Wnew=zeros(n,m); % initialize new weight vectors
        for i=1:m
            d=min(abs(i-winner),abs(40+i-winner)); % distance between node winning node in 1-D vector position
            u=exp(-d^2/(2*sigma^2)); % weight function for learning depends on distance from training city
            Wnew(:,i)=Wold(:,i)+u*eta*(xtrain-Wold(:,i));
        end
        eta=eta*alpha; % reduce learning rate
        sigma=sigma*alpha; % reduce width parameter
        error=norm(Wnew(:,winner)-Wold(:,winner)); % error is difference between old and new winning weight vectors
    end
    m=Wnew;
end

function showSOM(m) % displays SOM for RGB colors
    [l w K]=size(m);
    hold on
    for i=1:w
        for j=1:l
            scatter(j,w-i,1000,[m(i,j,1)/255 m(i,j,2)/255 m(i,j,3)/255],'filled','s');
        end
    end
    hold off
end

function showSOMWinners(m,winningNodes) % displays SOM for RGB colors
    [l w K]=size(m);
    hold on
    for i=1:w
        for j=1:l
            winner=0; % determine if current node was a winner.  0 is loser, 1 is winner
            for n=1:length(winningNodes)
                if winningNodes(n,1)==i && winningNodes(n,2)==j
                    winner=1;
                end
            end
            if winner==1 % plot winner's color
                scatter(j,w-i,1000,[m(i,j,1)/255 m(i,j,2)/255 m(i,j,3)/255],'filled','s');
            else % leave losers black
                scatter(j,w-i,1000,[0 0 0],'filled','s');
            end
        end
    end
    hold off
end

function X=SOMColorData(N) % generates N^3 RGB color vectors with N intensity levels for R, G, and B (0 to 255)
    X=zeros(N^3,3);
    intensities=round(linspace(0,255,N));
    for r=1:N
        for g=1:N
            for b=1:N
                X((N^2)*(r-1)+N*(g-1)+b,:)=[intensities(r) intensities(g) intensities(b)];
            end
        end
    end
    % filter out white, grey, and black
    Xfiltered=[];
    for i=1:N^3
        if X(i,1)~=X(i,2) || X(i,1)~=X(i,3) || X(i,2)~=X(i,3)
            Xfiltered=[Xfiltered; X(i,:)];
        end
    end
    X=Xfiltered;
end

function showImage(mk) % displays image for handwritten digits
    hold on
    for i=1:16
        for j=1:16
            val=1-((255-mk(i,j))/255);
            scatter(i,17-j,1000,[val val val],'filled','s');
        end
    end
    hold off
end

function Cm=ConfusionMatrix(y,yp)
    N=length(y);            % number of samples
    K=length(unique(y));    % number of classes
	Cm=zeros(K);            % initialize confusion matrix
    for n=1:N
        Cm(y(n),yp(n))=Cm(y(n),yp(n))+1; % fill in confusion matrix
    end
    %Cm
    %er=trace(Cm)/sum(sum(Cm)) % er=1 means 0% error.  All classifications are correct
end