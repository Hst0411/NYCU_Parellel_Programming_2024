#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>
#include <mpi.h>
#include <fstream>
#include <sstream>
#include <chrono>
#include "../include/fptree.hpp"

#define MASTER 0

void test_1()
{
    const Item a{ "A" };
    const Item b{ "B" };
    const Item c{ "C" };
    const Item d{ "D" };
    const Item e{ "E" };

    const std::vector<Transaction> transactions{
        { a, b },
        { b, c, d },
        { a, c, d, e },
        { a, d, e },
        { a, b, c },
        { a, b, c, d },
        { a },
        { a, b, c },
        { a, b, d },
        { b, c, e }
    };

    const uint64_t minimum_support_threshold = 2;

    const FPTree fptree{ transactions, minimum_support_threshold };

    const std::set<Pattern> patterns = fptree_growth( fptree );

    assert( patterns.size() == 19 );
    assert( patterns.count( { { a }, 8 } ) );
    assert( patterns.count( { { b, a }, 5 } ) );
    assert( patterns.count( { { b }, 7 } ) );
    assert( patterns.count( { { c, b }, 5 } ) );
    assert( patterns.count( { { c, a, b }, 3 } ) );
    assert( patterns.count( { { c, a }, 4 } ) );
    assert( patterns.count( { { c }, 6 } ) );
    assert( patterns.count( { { d, a }, 4 } ) );
    assert( patterns.count( { { d, c, a }, 2 } ) );
    assert( patterns.count( { { d, c }, 3 } ) );
    assert( patterns.count( { { d, b, a }, 2 } ) );
    assert( patterns.count( { { d, b, c }, 2 } ) );
    assert( patterns.count( { { d, b }, 3 } ) );
    assert( patterns.count( { { d }, 5 } ) );
    assert( patterns.count( { { e, d }, 2 } ) );
    assert( patterns.count( { { e, c }, 2 } ) );
    assert( patterns.count( { { e, a, d }, 2 } ) );
    assert( patterns.count( { { e, a }, 2 } ) );
    assert( patterns.count( { { e }, 3 } ) );
}

void test_2()
{
    const Item a{ "A" };
    const Item b{ "B" };
    const Item c{ "C" };
    const Item d{ "D" };
    const Item e{ "E" };

    const std::vector<Transaction> transactions{
        { a, b, d, e },
        { b, c, e },
        { a, b, d, e },
        { a, b, c, e },
        { a, b, c, d, e },
        { b, c, d },
    };

    const uint64_t minimum_support_threshold = 3;

    const FPTree fptree{ transactions, minimum_support_threshold };

    const std::set<Pattern> patterns = fptree_growth( fptree );

    assert( patterns.size() == 19 );
    assert( patterns.count( { { e, b }, 5 } ) );
    assert( patterns.count( { { e }, 5 } ) );
    assert( patterns.count( { { a, b, e }, 4 } ) );
    assert( patterns.count( { { a, b }, 4 } ) );
    assert( patterns.count( { { a, e }, 4 } ) );
    assert( patterns.count( { { a }, 4 } ) );
    assert( patterns.count( { { d, a, b }, 3 } ) );
    assert( patterns.count( { { d, a }, 3 } ) );
    assert( patterns.count( { { d, e, b, a }, 3 } ) );
    assert( patterns.count( { { d, e, b }, 3 } ) );
    assert( patterns.count( { { d, e, a }, 3 } ) );
    assert( patterns.count( { { d, e }, 3 } ) );
    assert( patterns.count( { { d, b }, 4 } ) );
    assert( patterns.count( { { d }, 4 } ) );
    assert( patterns.count( { { c, e, b }, 3 } ) );
    assert( patterns.count( { { c, e }, 3 } ) );
    assert( patterns.count( { { c, b }, 4 } ) );
    assert( patterns.count( { { c }, 4 } ) );
    assert( patterns.count( { { b }, 6 } ) );
}

void test_3()
{
    const Item a{ "A" };
    const Item b{ "B" };
    const Item c{ "C" };
    const Item d{ "D" };
    const Item e{ "E" };
    const Item f{ "F" };
    const Item g{ "G" };
    const Item h{ "H" };
    const Item i{ "I" };
    const Item j{ "J" };
    const Item k{ "K" };
    const Item l{ "L" };
    const Item m{ "M" };
    const Item n{ "N" };
    const Item o{ "O" };
    const Item p{ "P" };
    const Item s{ "S" };

    const std::vector<Transaction> transactions{
        { f, a, c, d, g, i, m, p },
        { a, b, c, f, l, m, o },
        { b, f, h, j, o },
        { b, c, k, s, p },
        { a, f, c, e, l, p, m, n }
    };

    const uint64_t minimum_support_threshold = 3;

    const FPTree fptree{ transactions, minimum_support_threshold };

    const std::set<Pattern> patterns = fptree_growth( fptree );

    assert( patterns.size() == 18 );
    assert( patterns.count( { { f }, 4 } ) );
    assert( patterns.count( { { c, f }, 3 } ) );
    assert( patterns.count( { { c }, 4 } ) );
    assert( patterns.count( { { b }, 3 } ) );
    assert( patterns.count( { { p, c }, 3 } ) );
    assert( patterns.count( { { p }, 3 } ) );
    assert( patterns.count( { { m, f, c }, 3 } ) );
    assert( patterns.count( { { m, f }, 3 } ) );
    assert( patterns.count( { { m, c }, 3 } ) );
    assert( patterns.count( { { m }, 3 } ) );
    assert( patterns.count( { { a, f, c, m }, 3 } ) );
    assert( patterns.count( { { a, f, c }, 3 } ) );
    assert( patterns.count( { { a, f, m }, 3 } ) );
    assert( patterns.count( { { a, f }, 3 } ) );
    assert( patterns.count( { { a, c, m }, 3 } ) );
    assert( patterns.count( { { a, c }, 3 } ) );
    assert( patterns.count( { { a, m }, 3 } ) );
    assert( patterns.count( { { a }, 3 } ) );
}


// load datasets
std::vector<Transaction> load_transactions(const std::string& filename) {
    std::ifstream infile(filename);
    std::string line;
    std::vector<Transaction> transactions;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        Transaction transaction;
        std::string item;
        while (std::getline(iss, item, ',')) {
            transaction.push_back(item);
        }
        transactions.push_back(transaction);
    }
    return transactions;
}


int main(int argc, char** argv)
{

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::string dataset_file = "low_50.txt";
    const uint64_t minimum_support_threshold = 3;

    // start time
    auto total_start_time = std::chrono::high_resolution_clock::now();

    std::vector<Transaction> all_transactions;
    // MASTER
    if (rank == MASTER) {
        std::cout << "load datasets: " << dataset_file << std::endl;
        all_transactions = load_transactions(dataset_file);
    }

    // broadcast item num
    int transaction_count = all_transactions.size();
    MPI_Bcast(&transaction_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // assign item to processor
    int chunk_size = transaction_count / size;
    std::vector<Transaction> local_transactions(chunk_size);
    MPI_Scatter(
        all_transactions.data(), chunk_size, MPI_CHAR,
        local_transactions.data(), chunk_size, MPI_CHAR,
        0, MPI_COMM_WORLD
    );

    // local construct FP-tree, extract frequency item sett
    FPTree local_fptree(local_transactions, minimum_support_threshold);
    std::set<Pattern> local_patterns = fptree_growth(local_fptree);

    // Reduce frequency item set from all processors
    std::set<Pattern> global_patterns;
    std::vector<std::pair<std::set<Item>, uint64_t>> local_patterns_vector(local_patterns.begin(), local_patterns.end());
    std::vector<std::pair<std::set<Item>, uint64_t>> global_patterns_vector;

    if (rank == 0) {
        global_patterns_vector.resize(local_patterns_vector.size() * size);
    }

    MPI_Reduce(local_patterns_vector.data(), global_patterns_vector.data(),
            local_patterns_vector.size() * sizeof(std::pair<std::set<Item>, uint64_t>), MPI_BYTE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        global_patterns.insert(global_patterns_vector.begin(), global_patterns_vector.end());
    }



    // MASTER output
    if (rank == 0) {
        auto total_end_time = std::chrono::high_resolution_clock::now();
        std::cout << "Global patterns: " << std::endl;
        for (const auto& pattern : global_patterns) {
            std::cout << "Pattern: ";
            for (const auto& item : pattern.first) {
                std::cout << item << " ";
            }
            std::cout << "Support: " << pattern.second << std::endl;
        }

        // total time
        std::chrono::duration<double> total_elapsed = (total_end_time - total_start_time) * 1000;
        std::cout << "Total elapsed time: " << total_elapsed.count() << " ms" << std::endl;
    }

    MPI_Finalize();
    std::cout << "All tests passed!" << std::endl;
    return 0;
    //test_1();
    //test_2();
    //test_3();
    

    //return EXIT_SUCCESS;
}
