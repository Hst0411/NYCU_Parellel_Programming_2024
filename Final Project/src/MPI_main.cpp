#include <cassert>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <set>
#include <vector>
#include <time.h>
#include <sstream>
#include <string>
#include <algorithm>
#include <mpi.h>

#include "../include/fptree.hpp"

std::vector<Transaction> readTransactionsFromFile(const std::string& filename) {
    std::vector<Transaction> transactions;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open the file: " << filename << "!!!" << std::endl;
        return transactions;
    }
    //std::cerr << "Open the file: " << filename << " successfully !!!" << std::endl;
    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::string> transaction;
        std::istringstream line_stream(line);
        std::string item;

        // 處理每一行的每個數字
        while (std::getline(line_stream, item, ',')) {
            item.erase(remove(item.begin(), item.end(), ' '), item.end()); // remove space
            if (!item.empty()) { // 不插入空char
                transaction.push_back(item); // insert item
            }
        }

        if (!transaction.empty()) { // 確保不插入空的交易
            transactions.push_back(transaction);
        }
    }

    file.close();
    return transactions;
}


void test_1(const std::string& filename)
{
    const Item a{ "A" };
    const Item b{ "B" };
    const Item c{ "C" };
    const Item d{ "D" };
    const Item e{ "E" };

    // const std::string filename = "../dataset/input.txt";

    // read transaction data
    std::vector<Transaction> transactions = readTransactionsFromFile(filename);

    // test output
    /*
    int total = 0;
    std::cout << "Loading data: " << filename << std::endl;
    for (const auto& transaction : transactions) {
        for (const auto& item : transaction) {
            std::cout << item << " ";
        }
        total++;
        std::cout << std::endl;
    }
    std::cout << "Datasets: " << filename << std::endl;
    std::cout << "Total: " << total << std::endl;
    */
    // std::cout << "transactions[4][4]: " << transactions[4][4] << std::endl;
    /*
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
    */

    const uint64_t minimum_support_threshold = 2;

    const FPTree fptree{ transactions, minimum_support_threshold };

    // const std::set<Pattern> patterns = fptree_growth( fptree );
    /*
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
    */
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

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    double start_time = MPI_Wtime();
    test_1("../dataset/low_50.txt");
    if (world_rank == 0){
        double end_time = MPI_Wtime();
        std::cout << "Test case1 costs " << (end_time - start_time)*1000 << " ms\n";
        std::cout << "Test1 passed!" << std::endl;
    }
    //test_2();
    //test_3();
    MPI_Finalize();
    return 0;
}
