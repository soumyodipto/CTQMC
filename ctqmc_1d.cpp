#include <iostream>
#include <random>
#include <cmath>
#include <string>
#include <tuple>
#include <stdlib.h>
#include <algorithm>
#include <armadillo>
#include <ctime>
using namespace std;
using namespace arma;




const double pi = 3.14159;
// Define global variables
double mass = 1.0;
double hbar = 1.0;
double k_b = 1.0;
double lam = 0.05; // V = -V0*exp(-\lambda x^2)
double V0 = 0.5/lam;
double T = 0.1;
double beta = 1/(k_b*T);
double l = 100.0;

// Define global MC variables
int numMCsteps = 100000;
int numEquil = numMCsteps/5;
int numObsInterval = 5;

//Define Z0, NegDervZ0
double Z0 = pow(mass/(2*pi*hbar*beta), 0.5)*l;
double NegDervZ0 = 0.5/beta*Z0;


tuple<vec, vec> ABs(vec all_ts, int order)
{
    vec all_As(order);
    vec all_Bs(order);
    for(int i=0; i < order; i++)
    {
        all_As(i) = pow(mass/(2*pi*hbar*all_ts(i)), 0.5);
        all_Bs(i) = 0.5*mass/(all_ts(i)*hbar);
    }
    return make_tuple(all_As, all_Bs);
}



vec GetAll_ts(vec all_taus, int order)
{
    vec all_ts(order);
    all_ts(0) = beta - all_taus(0) + all_taus(order-1);
    for(int i=1; i < order; i++)
    	all_ts(i) = all_taus(i-1) - all_taus(i);
    return all_ts;        
}



mat Make_M_matrix(vec all_Bs, int order)
{
    mat M(order, order, fill::zeros);
    if(order == 1)
        M(0,0) = 2*lam;
    else if(order == 2)
    {
        M(0,0) = 2*(all_Bs(0) + all_Bs(1) + lam);   
	M(0,1) = -2*(all_Bs(0)+all_Bs(1));
	M(1,0) = M(0,1);
	M(1,1) = M(0,0);
    }
    else
    {
    	for(int i=0; i < order-1; i++)
    	{
    	    M(i,i) = 2*(all_Bs(i) + all_Bs(i+1) + lam);
    	    M(i,i+1) = -2*all_Bs(i+1);
    	    M(i+1,i) = M(i,i+1);    	    
    	}
    	M(order-1,order-1) = 2*(all_Bs(0) + all_Bs(order-1) + lam);
        M(0,order-1) = -2*all_Bs(0);
        M(order-1,0) = M(0,order-1);	
    }
    return M;
}


double DenConfigWeight(vec all_As, mat M, int order)
{
    double fac1 = 1.0;
    for(int i = 0; i < order; i++)
	fac1 *= all_As[i];
    double detM = det(M/(2*pi));
    return fac1/pow(detM,0.5);
}




double PhiConfigWeight(vec taus, int order)
{
    taus.shed_row(0);
    vec val_add(1, fill::ones);
    val_add *= val_add;	
    taus.insert_rows(0, val_add);
    vec all_ts = GetAll_ts(taus, order);
    tuple<vec,vec> AB = ABs(all_ts, order);
    vec all_As = get<0>(AB);
    vec all_Bs = get<1>(AB);
    mat M = Make_M_matrix(all_Bs, order);
    double phi = DenConfigWeight(all_As, M, order);
    return phi;
}


double PsiConfigWeight(vec all_ts, vec all_As, vec all_Bs, mat M, int order)
{
    mat Minv = inv_sympd(M);
    double fac1 = DenConfigWeight(all_As, M, order);
    double fac2 = (-0.5 + all_Bs[0]*(Minv(0,0)+Minv(order-1,order-1)-2*Minv(0,order-1)));
    return fac1*fac2/all_ts[0];
}








tuple<double,double> NumConfigWeight(vec taus, mat M, int order, vec all_ts, vec all_As, vec all_Bs)
{
    double phi;
    double psi;
    if(order == 0)
    {
	phi = 0;
	psi = NegDervZ0;
    }
    else
    {
        phi = -PhiConfigWeight(taus, order);
	psi = -PsiConfigWeight(all_ts, all_As, all_Bs, M, order);
    }
    return make_tuple(phi, psi);
}















vec GetNewTaus(vec taus, int order_old, string MCmove)
{
    if(MCmove == "insert")
    {
        vec val_add = beta*randu<vec>(1);
        taus.insert_rows(0, val_add);
	taus = sort(taus, "descend");        
    }
    else if(MCmove == "remove")
    {
	vec rem_indvec = randi<vec>(1, distr_param(0,order_old-1));
        int rem_ind = rem_indvec(0);
        taus.shed_row(rem_ind);
    }
    else
    {
	vec rem_indvec = randi<vec>(1, distr_param(0,order_old-1));
	int rem_ind = rem_indvec(0);
        taus.shed_row(rem_ind);
        vec val_add = beta*randu<vec>(1);
	taus.insert_rows(0, val_add);
	taus = sort(taus, "descend");
    }
    return taus;
}



tuple<int,vec,vec,vec,vec,mat,double> GetNewFeatures(vec taus_old, int order_old, int order_new, string MCmove)
{
    vec taus_new = GetNewTaus(taus_old, order_old, MCmove);
    vec all_ts_new = GetAll_ts(taus_new, order_new);
    tuple<vec,vec> AB_new = ABs(all_ts_new, order_new);
    vec all_As_new = get<0>(AB_new);
    vec all_Bs_new = get<1>(AB_new);
    mat M_new = Make_M_matrix(all_Bs_new, order_new);
    double den_configweight_new = DenConfigWeight(all_As_new, M_new, order_new);
    return make_tuple(order_new, taus_new, all_ts_new, all_As_new, all_Bs_new, M_new, den_configweight_new);
        
}



tuple<int,vec,vec,vec,vec,mat,double> PackNewFeatures(vec taus_old, int order_old, string MCmove)
{
    //Return empty arrays and Z0 if order_new=0
    int order_new;
    vec taus_new;
    vec all_ts_new;
    vec all_As_new;
    vec all_Bs_new;
    mat M_new;
    double den_configweight_new = Z0;
    if(MCmove == "insert")
        order_new = order_old+1;
    else if(MCmove == "remove")
        order_new = order_old-1;
    else
        order_new = order_old;

    if(order_new == 0)
        return make_tuple(order_new, taus_new, all_ts_new, all_As_new, all_Bs_new, M_new, den_configweight_new);
    else
    {
	tuple<int,vec,vec,vec,vec,mat,double> new_features = GetNewFeatures(taus_old, order_old, order_new, MCmove);
        return new_features;
    }
}



bool MCMoveDecision(int order_old, int order_new, double configweight_old, double configweight_new, string MCmove)
{
    bool decision = false;
    vec rnum_vec = randu<vec>(1);
    double rnum = rnum_vec(0);
    if(MCmove == "insert")
    {
	double R_n_nplus1 = V0*configweight_new*beta/configweight_old/order_new;
	if(R_n_nplus1 >= 1.0)
	    decision = true;
	else if(R_n_nplus1 < 1.0 && R_n_nplus1 > rnum)
	    decision = true;
    }
    
    if(MCmove == "remove")
    {
	double R_n_nplus1 = V0*configweight_old*beta/configweight_new/order_old;
        double R_nplus1_n = 1/R_n_nplus1;
	if(R_nplus1_n >= 1.0)
	    decision = true;
	else if(R_nplus1_n < 1.0 && R_nplus1_n > rnum)
	    decision = true;
    }

    if(MCmove == "stay")
    {
	double R = configweight_new/configweight_old;
	if(R >= 1.0)
	    decision = true;
	else if(R < 1.0 && R > rnum)
	    decision = true;
    }
    return decision;
}







void CTQMC_MarkovChain()
{
    string MCmove = "";
    int timeOrder_new;	
    vec taus_new;
    vec all_ts_new;
    vec all_As_new;
    vec all_Bs_new;
    mat M_new;
    double den_configweight_new;

    int timeOrder_tmp;	
    vec taus_tmp;
    vec all_ts_tmp;
    vec all_As_tmp;
    vec all_Bs_tmp;
    mat M_tmp;
    double den_configweight_tmp;

    double num_trace[(numMCsteps - numEquil)/numObsInterval];
    int order_trace[(numMCsteps - numEquil)/numObsInterval];

    int timeOrder_start = 1;	//define initial NON-ZERO timeOrder
    int timeOrder_old = timeOrder_start;
    arma_rng::set_seed_random();  // set the seed to a random value
    vec taus_old = beta*randu<vec>(timeOrder_old);
    taus_old = sort(taus_old, "descend");

    vec all_ts_old = GetAll_ts(taus_old, timeOrder_old);
    tuple<vec,vec> AB_old = ABs(all_ts_old, timeOrder_old);
    vec all_As_old = get<0>(AB_old);
    vec all_Bs_old = get<1>(AB_old);
    
    mat M_old = Make_M_matrix(all_Bs_old, timeOrder_old);
    
    double den_configweight_old = DenConfigWeight(all_As_old, M_old, timeOrder_old);

    int obs_count = 0;

    for(int step=0; step < numMCsteps; step++)
    {
	//cout << timeOrder_old << endl;
        //Selects insert, remove or stay move
	if(timeOrder_old == 0)
	{
	    string moves[2] = {"insert","stay"};
            vec indvec = randi<vec>(1, distr_param(0,1));
            int ind = indvec(0);
	    MCmove = moves[ind]; 
	}
	else
	{
	    string moves[3] = {"insert","remove","stay"};
            vec indvec = randi<vec>(1, distr_param(0,2));
            int ind = indvec(0);
	    MCmove = moves[ind];
	}   
	
	//Action based on the move selected
        tuple<int,vec,vec,vec,vec,mat,double> all_new_features = PackNewFeatures(taus_old, timeOrder_old, MCmove);
	// New configuratuion features after making a MC move
        timeOrder_new = get<0>(all_new_features);
        taus_new = get<1>(all_new_features);
	all_ts_new = get<2>(all_new_features);
	all_As_new = get<3>(all_new_features);
	all_Bs_new = get<4>(all_new_features);
	M_new = get<5>(all_new_features);
	den_configweight_new = get<6>(all_new_features);

        //Check if MC move is accepted or not
	bool MoveDecision = MCMoveDecision(timeOrder_old, timeOrder_new, den_configweight_old, den_configweight_new, MCmove);

	//Update variables based on acceptance
	if(MoveDecision)
	{
	    timeOrder_tmp = timeOrder_new;	
            taus_tmp = taus_new;
	    all_ts_tmp = all_ts_new;
	    all_As_tmp = all_As_new;
	    all_Bs_tmp = all_Bs_new;
	    M_tmp = M_new;
	    den_configweight_tmp = den_configweight_new;
	}
	else
	{
	    timeOrder_tmp = timeOrder_old;	
            taus_tmp = taus_old;
	    all_ts_tmp = all_ts_old;
	    all_As_tmp = all_As_old;
	    all_Bs_tmp = all_Bs_old;
	    M_tmp = M_old;
	    den_configweight_tmp = den_configweight_old;
	}
	
	//Collect observables
	if(step > numEquil && step % numObsInterval == 0)
	{
	    order_trace[obs_count] = timeOrder_tmp;
	    tuple<double,double> num_phi_psi = NumConfigWeight(taus_tmp, M_tmp, timeOrder_tmp, all_ts_tmp, all_As_tmp, all_Bs_tmp);
            double num_phi = get<0>(num_phi_psi);
	    double num_psi = get<1>(num_phi_psi);
	    double obs_num = timeOrder_tmp*(timeOrder_tmp/beta*num_phi + num_psi)/den_configweight_tmp;
	    num_trace[obs_count] = obs_num;
	    obs_count += 1;
	
	}

        //Update config for next iteration
        timeOrder_old = timeOrder_tmp;	
        taus_old = taus_tmp;
        all_ts_old = all_ts_tmp;
        all_As_old = all_As_tmp;
        all_Bs_old = all_Bs_tmp;
        M_old = M_tmp;
        den_configweight_old = den_configweight_tmp;     
    }
    for(int i=0; i < (numMCsteps - numEquil)/numObsInterval - 1; i++)
    {
	if(order_trace[i] > 200)
	    cout << "Ping" << endl;
    }    
}













int main()
{
    int start_s=clock();
    cout << "Temperature = " << T << endl;
    cout << "Lambda = " << lam << endl;
    
    CTQMC_MarkovChain();
    int stop_s=clock();
    cout << "time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC) << endl;

	
    
    return 0;
}
