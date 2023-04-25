#include "common/kmer.hpp"
#include "common/alphabets.hpp"
#include "index/kmer_index.hpp"
#include "mxx/env.hpp"
#include <bits/stdc++.h>
#include <mpi.h>
#include<omp.h>
#include "kmerhash/distributed_robinhood_map.hpp"
#include "kmerhash/distributed_batched_robinhood_map.hpp"

//set K-value and testcase file name
const int K_value = 28;
const int M_value = 9;

// string test_file_name = "/home/adarsh.dharmadevan/datasets/FV/SRR072005_12.fastq";
string test_file_name = "/home/adarsh.dharmadevan/datasets/FV/SRR072005_10000.fastq";
//string test_file_name = "/home/adarsh.dharmadevan/datasets/FV/merged.fastq";


//----------------Type definations for kmer index-----------------------------------
template <typename key>
using map_parm = bliss::index::kmer::SingleStrandHashMapParams<key>;

template <typename tupple_type>
using fastq_paser = bliss::io::FASTQParser<tupple_type>;

template<typename Iterator, template<typename> class Parser>
using seq_iter = bliss::io::SequencesIterator<Iterator, Parser>;


typedef bliss::common::alphabet::DNA_T<> alph;

typedef bliss::common::Kmer<K_value, alph> KmerType;
typedef bliss::common::Kmer<M_value, alph> MinimizerType; //sample lexicographic minimizer class for testing
typedef vector<char> SupermerType;

// using MapType = dsc::counting_unordered_map<KmerType, long unsigned int, map_parm>;
// using MapType = dsc::counting_robinhood_map<KmerType, long unsigned int, map_parm>; 
using MapType = dsc::minimizer_based_counting_unordered_map< std::tuple<MinimizerType, SupermerType >, KmerType, long unsigned int, map_parm>; 


// typedef bliss::index::kmer::CountIndex2<MapType> kmer_index;
typedef bliss::index::kmer::MyIndex<MapType, MinimizerType> kmer_index;

// typedef std::pair<std::vector<std::string>, std::vector<int>> MykmerAndCountsVec;
typedef std::pair<std::vector<std::string>, std::vector<long unsigned int>> MykmerAndCountsVec;

// typedef std::pair<KmerType, unsigned long int> MyKmerCountPair;
typedef std::pair<KmerType, long unsigned int> MyKmerCountPair;

typedef decltype(::std::declval<MapType>().count(::std::declval<std::vector<KmerType> &>())) kmerAndCountsVec;

kmerAndCountsVec just;
typedef decltype(just[0]) KmerCountPair;
//-----------------------------------------------------------------------------------


//-----------------------Function declarations---------------------------------------
bool test_for_fastq(std::string file_name, mxx::comm& comm);

bool equal_my(kmerAndCountsVec counts, MykmerAndCountsVec file);

MykmerAndCountsVec read_kmer_counts(std::string file_name);

std::vector<KmerType> get_kmerind_kmers(std::vector<std::string> file_string);

bool less_than_sort(KmerCountPair i, KmerCountPair j);

bool less_than_lower(KmerCountPair i, MyKmerCountPair j);

int getNodeCount(void)
{
   int rank, is_rank0, nodes;
   MPI_Comm shmcomm;

   MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                       MPI_INFO_NULL, &shmcomm);
   MPI_Comm_rank(shmcomm, &rank);
   is_rank0 = (rank == 0) ? 1 : 0;
   MPI_Allreduce(&is_rank0, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Comm_free(&shmcomm);
   return nodes;
}
//-----------------------------------------------------------------------------------



int main(int argc, char** argv)
{
    //OpenMPI intialization
    mxx::env e(argc, argv);
    mxx::comm comm;
	int n = getNodeCount();
    if (comm.rank() == 0)   
    {
        printf("EXECUTING %s\n", argv[0]);

        // print the number of mpi processes and the number of nodes
        int MPI_Comm_size = comm.size();
        int MPI_Comm_rank = comm.rank();
        std::cout << "MPI_COMM_SIZE : " << MPI_Comm_size << std::endl;
        std::cout << "MPI_COMM_RANK : " << MPI_Comm_rank << std::endl;

        // print node for each process
        // {
        //     volatile int i = 0;
        //     char hostname[256];
        //     gethostname(hostname, sizeof(hostname));
        //     printf("PID %d on %s ready for attach\n", getpid(), hostname);
        //     fflush(stdout);
        //     while (0 == i)
        //         sleep(5);
        // }
    }


    //comm.barrier(); // Be carefull


    //-------------correctness test----------------

    kmer_index first(comm);
    first.build_mpiio<fastq_paser, seq_iter>(test_file_name, comm);
    if(test_for_fastq(test_file_name, comm) == true)
        std::cout << "Ran all test cases successfully" << std::endl;
    else 
        std::cout << "Test cases failed" << std::endl;
    
    //----------------------------------------------


    //-------------time measurement-----------------
    
    // auto begin = std::chrono::high_resolution_clock::now();    

    // //KmerIndex creation
    // kmer_index first(comm);
    // first.build_mpiio<fastq_paser, seq_iter>(test_file_name, comm);
    
    // auto end = std::chrono::high_resolution_clock::now();
    // std::cout << "Time taken for building index : " << std::chrono::duration<double>(end-begin).count() << std::endl;
    
    //----------------------------------------------
  
    return 0;
}







//Function definations
bool test_for_fastq(std::string file_name, mxx::comm& comm)
{
    MykmerAndCountsVec actual_counts = read_kmer_counts(file_name);
    std::vector<KmerType> kmerind_kmers = get_kmerind_kmers(actual_counts.first);

    //KmerIndex creation
    kmer_index first(comm);
    first.build_mmap<fastq_paser, seq_iter>(file_name, comm);
    auto counts = first.find(kmerind_kmers);
    
    bool passed = equal_my(counts, actual_counts);
    return passed;
}


bool equal_my(kmerAndCountsVec counts, MykmerAndCountsVec file_data)
{
    std::sort(counts.begin(), counts.end(), less_than_sort);
    std::vector<long unsigned int> file_counts = file_data.second;
    std::vector<KmerType> file_kmers = get_kmerind_kmers(file_data.first);

    bool equal = true;    
    bool i_correct = true;
    for(int i = 0; i < file_counts.size(); i++)
    {
        MyKmerCountPair val(file_kmers[i], file_counts[i]);
        auto loc = std::lower_bound(counts.begin(), counts.end(), val, less_than_lower);
        if(loc != counts.end())
        {
            KmerCountPair match = *loc;
            if(match.first == file_kmers[i] && match.second == file_counts[i])
                i_correct = true;
            else
                i_correct = false;
        }
        else
            i_correct = false;


        if(i_correct == false)
        {
            equal = false;
            break;
        }
    } 

    return equal;
}


MykmerAndCountsVec read_kmer_counts(std::string file_name)
{
    std::string in_file = file_name + ".counts";
    std::ifstream file_stream(in_file);

    
    std::vector <std::string> file_kmers;  
    std::vector <long unsigned int> file_counts;
    std::string line;
    while (getline(file_stream, line))
    {
        //Where to split
        int pos = line.find("@");

        //extracting kmer and number
        std::string kmer = line.substr(0, pos); // store the substring   
        line.erase(0, pos + 1);  

        //adding kmer and number to function
        file_kmers.push_back(kmer);
        file_counts.push_back(stoi(line));
    }
    
    return std::make_pair(file_kmers, file_counts);
}



std::vector<KmerType> get_kmerind_kmers(std::vector<std::string> file_string)
{
    std::vector<KmerType> kmerind_kmers;
    for(int i = 0; i < file_string.size(); i++)
    {
        kmerind_kmers.push_back(KmerType(file_string[i]));
    }
    return kmerind_kmers;
}


bool less_than_sort(KmerCountPair i, KmerCountPair j)
{
    return (i.first < j.first);
}


bool less_than_lower(KmerCountPair i, MyKmerCountPair j)
{
    return i.first < j.first;
}
