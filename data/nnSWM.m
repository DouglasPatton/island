function S=nnSWM(xcoord,ycoord,N,Contig,RowStdOpt)
% PURPOSE: uses x,y coordinates to produce a nearest-neighbor spatial
%          weight matrix. The user is asked to input x and y coordinates,
%          as well as the number of neighbors. Several configuration
%          options are available for the specification of the individual
%          elements of the SWM.
% ------------------------------------------------------
% USAGE: W = nnSWM(xcoord,ycoord,N,Contig,Rowstdopt)
% where:     xcoord = x-direction coordinate
%            ycoord = y-direction coordinate
%            N = number of neighbors to include in the SWM
%            Contig = 1 for individual SWM(i,j) entries = 1
%                   = 0 for individual SWM(i,j) entries = inverse distance
%            rowstdopt = 1 for row-standardization
%                      = 0 for no standardization
% ------------------------------------------------------
% RETURNS: 
%          S = a sparse weight matrix based on the given number of
%              neighbors. A 1 is entered in SWM(i,j) if i and j are 
%              deemed neighbors (unless contig=0, where inverse distance
%              appears for the designated n neighbors.
% ------------------------------------------------------
%
% written by: Patrick Walsh
% Walsh.Patrick@epa.gov
% Economist
% National Center for Environmental Economics, US EPA

% First, preallocate a matrix for the nearest neighbor results
Neighbors = zeros(length(xcoord)*N,3);

% Start looping over all observations
for i = 1:length(xcoord)
	% Create a matrix to put all the distances between i and j and i and j
	DistMat = zeros(length(xcoord),3);
	
	% Start loop between i and all other j
	for j=1:length(xcoord)
		% Calculate distance between observations
  		Xdist=abs(xcoord(i)-xcoord(j));
		Ydist=abs(ycoord(i)-ycoord(j));
		D = sqrt(Xdist^2 + Ydist^2);
  		DistMat(j,:) = [i j D];
	end
	% Now that we have the matrix of distances and is and js, want to take the nearest
	% N neighbors by distance,  excluding the self distance (i==j)
	% First, find i==j and remove it (don't want an observation to be a neighbor to itself)
	A = find(DistMat(:,1)==DistMat(:,2));
	DistMat(A,:)=[];
	% Next sort the distance matrix
	DistMatSort = sortrows(DistMat,3);
	% Keep the closest N items
	Nneighbors = DistMatSort(1:N,:);
	% this matrix contains the N nearest neighbors
	
	% If Contig == 1, replace the distances with equal weight
	if Contig == 1;
		Nneighbors(:,3) = ones(N,1);
	else 
		% replace any zeros in the distance matrix to -1 (so that division does not crap out)
		Dists = Nneighbors(:,3);
		Dists(Dists==0)=-1;
		Dists = 1./Dists;
		% bring back to zero
		Dists(Dists<0)=0;
		Nneighbors(:,3) = Dists;
	end
	% If indicated, standardize the weights matrix
	if RowStdOpt == 1;
		Nneighbors(:,3) = Nneighbors(:,3)./sum(Nneighbors(:,3));
	end
	% Put this set of N observations into the preallocated matrix.
	Neighbors(((i-1)*N+1):i*N,:) = Nneighbors;
end
n=length(xcoord);
S = sparse(Neighbors(:,1),Neighbors(:,2),Neighbors(:,3),n,n);
	


