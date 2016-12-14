/*
 *	PURPOSE:	To explore the effect of various matrix implementations on the speed of matrix multiplication.
 *			Currently implemented are c-style arrays, c++ std::vectors, and blitz arrays. The code
 *			reports on single-thread speed for NxN matrix multiplication, and scaling behaviour
 *			with naive openMP parallelisation.
 *
 *	REQUIRES:	Blitz++, OpenMP
 *
 *	AUTHOR: 	Murray Cutforth
 *
 *	DATE:		14/12/2016
 *
 *	TODO:		Test different ways of unrolling the matrix multiplication for loop
 */


#include <omp.h>
#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <blitz/array.h>



// Set matrix dimensions

static const int A_m = 1000;
static const int A_n = 1000;
static const int B_n = 1000;
static const int B_m = A_n;
static const int C_m = A_m;
static const int C_n = B_n;




// ========================================== CLASS DEFINITIONS ============================================//


class matrix_base {

	/*
	 *	Abstract base class for matrix storage container
	 */
	
	public:

	virtual double& operator() (int i, int j) =0;
};


class matrix_carray : public matrix_base {

	/*
	 *	Two dimensional c-style array implementation
	 */

	public:

	const int m;
	const int n;
	double** A;

	matrix_carray (int m_in, int n_in)
	:
		m	(m_in),
		n	(n_in)
	{
		A = new double*[m];

		for (int i=0; i<m; i++)
		{
			A[i] = new double[n];
		}
	}

	~matrix_carray()
	{
		for (int i=0; i<m; i++)
		{
			delete[] A[i];
		}

		delete[] A;
	}


	// Skip deep copy and deep move functions


	double& operator() (int i, int j)
	{
		return A[i][j];
	}
};



class matrix_blitz : public matrix_base {

	/*
	 * 	Blitz++ array implementation
	 */

	public:

	const int m;
	const int n;
	blitz::Array<double,2> A;

	matrix_blitz (int m_in, int n_in)
	:
		m	(m_in),
		n	(n_in),
		A	(m,n)
	{}

	double& operator() (int i, int j)
	{
		return A(i,j);
	}
};



class matrix_cppvec : public matrix_base {

	/*
	 *	Two dimensional c++ std::vector implementation
	 */

	public:

	const int m;
	const int n;
	std::vector<std::vector<double> > A;

	matrix_cppvec (int m_in, int n_in)
	:
		m	(m_in),
		n	(n_in),
		A	(m, std::vector<double>(n))
	{}

	double& operator() (int i, int j)
	{
		return A[i][j];
	}
};



//======================================= FUNCTION DEFINITIONS ============================================//


void write_matrices_to_file (

	std::shared_ptr<matrix_base> A,
	std::shared_ptr<matrix_base> B,
	std::shared_ptr<matrix_base> C
)
{
	/*
	 *	Write values of all three matrices to file in current directory
	 */
	
	
	std::cout << "Writing matrices to file..." << std::endl;
	
	std::ofstream outfileA;
	outfileA.open("A.dat");
	for (int i=0; i<A_m; i++)
	{
		for (int j=0; j<A_n; j++)
		{
			outfileA << (*A)(i,j) << " ";
		}
		outfileA << std::endl;
	}

	std::ofstream outfileB;
	outfileB.open("B.dat");
	for (int i=0; i<B_m; i++)
	{
		for (int j=0; j<B_n; j++)
		{
			outfileB << (*B)(i,j) << " ";
		}
		outfileB << std::endl;
	}
	
	std::ofstream outfileC;
	outfileC.open("C.dat");
	for (int i=0; i<C_m; i++)
	{
		for (int j=0; j<C_n; j++)
		{
			outfileC << (*C)(i,j) << " ";
		}
		outfileC << std::endl;
	}

	std::cout << "Output complete." << std::endl;
}


void construct_initialise (

	std::shared_ptr<matrix_base>& A,
	std::shared_ptr<matrix_base>& B,
	std::shared_ptr<matrix_base>& C,
	const int container_choice,
	const int Ainitchoice,
	const int Binitchoice
)
{
	/*
	 * Initialise the arrays using selected container type and initial values
	 */
	
	if (container_choice == 1)
	{
		A = std::make_shared<matrix_carray>(A_m,A_n);
		B = std::make_shared<matrix_carray>(B_m,B_n);
		C = std::make_shared<matrix_carray>(C_m,C_n);
	}
	else if (container_choice == 2)
	{
		A = std::make_shared<matrix_blitz>(A_m,A_n);
		B = std::make_shared<matrix_blitz>(B_m,B_n);
		C = std::make_shared<matrix_blitz>(C_m,C_n);
	}
	else if (container_choice == 3)
	{
		A = std::make_shared<matrix_cppvec>(A_m,A_n);
		B = std::make_shared<matrix_cppvec>(B_m,B_n);
		C = std::make_shared<matrix_cppvec>(C_m,C_n);
	}
	

	// Various options for the initial values of the matrices A and B have been implemented below

	std::cout << "Initialising A and B.." << std::endl;

	for (int i=0; i<A_m; i++)
	{
		for (int j=0; j<A_n; j++)
		{
			if (Ainitchoice == 1)
			{
				(*A)(i,j) = i+j;
			}
			else if (Ainitchoice == 2)
			{
				(*A)(i,j) = 1;
			}
		}
	}

	for (int i=0; i<B_m; i++)
	{
		for (int j=0; j<B_n; j++)
		{
			if (Binitchoice == 1)
			{
				(*B)(i,j) = i*j;
			}
			else if (Binitchoice == 2)	// Identity matrix
			{
				(*B)(i,j) = (i == j) ? 1.0 : 0.0;
			}
			else if (Binitchoice == 3)
			{
				(*B)(i,j) = (i+j)%(B_n/2);
			}
		}
	}
	
	for (int i=0; i<C_m; i++)
	{
		for (int j=0; j<C_n; j++)
		{
			(*C)(i,j) = 0.0;
		}
	}

	std::cout << "Initialisation complete." << std::endl;
};


double matrix_multiplication_time (

	std::shared_ptr<matrix_base> A,
	std::shared_ptr<matrix_base> B,
	std::shared_ptr<matrix_base> C,
	const int numthreads
)
{
	/*
	 * Perform matrix multiplication using given number of threads in OpenMP and return run time
	 */
	
	double starttime = omp_get_wtime();

	#pragma omp parallel for num_threads(numthreads) schedule(static)
	for (int i=0; i<A_m; i++)
	{
		for (int j=0; j<B_n; j++)
		{
			for (int k=0; k<A_n; k++)
			{
				(*C)(i,j) += (*A)(i,k) * (*B)(k,j);
			}
		}
	}

	double endtime = omp_get_wtime();
	double runtime = endtime - starttime;

	std::cout << '\t' << "Matrix multiplication finished using " << numthreads << " threads. Runtime = " << runtime << "s" << std::endl;

	return runtime;
}

	




// ========================================= MAIN FUNCTION =============================================== //


int main()
{
	// Initial conditions choice
	
	const int Ainitchoice = 2;
	const int Binitchoice = 3;
	const int maxnumthreads = 32;

	std::shared_ptr<matrix_base> A;
	std::shared_ptr<matrix_base> B;
	std::shared_ptr<matrix_base> C;


	// Output timing data to current directory

	std::ofstream outfile;
	outfile.open("timingresults.dat");

	for (int i=1; i<=maxnumthreads; i*=2) outfile << i << " ";
	outfile << std::endl;

	
	// Compare all containers

	for (int containerchoice = 1; containerchoice <= 3; containerchoice++)
	{
		construct_initialise(A,B,C,containerchoice,Ainitchoice,Binitchoice);

		std::cout << "Testing container choice " << containerchoice << "..." << std::endl;

		for (int i=1; i<=maxnumthreads; i*=2)
		{
			double runtime = matrix_multiplication_time(A,B,C,i);
			outfile << runtime << " ";
		}
		outfile << std::endl;

		std::cout << std::endl;
	}
	

	write_matrices_to_file(A,B,C);
}
