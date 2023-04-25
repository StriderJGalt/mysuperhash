/*
 * Copyright 2015 Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file    big_collective.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @group   collective
 * @brief   Implementations of big (> INT_MAX) collective operations.
 */

#ifndef MXX_BIG_COLLECTIVE_HPP
#define MXX_BIG_COLLECTIVE_HPP

#include <vector>

#include "datatypes.hpp"
#include "shift.hpp" // FIXME: include only `requests`

namespace mxx {
namespace impl {

/**
 * @brief   Returns the displacements vector needed by MPI_Alltoallv.
 *
 * @param counts    The `counts` array needed by MPI_Alltoallv
 *
 * @return The displacements vector needed by MPI_Alltoallv.
 */
template <typename index_t = int>
std::vector<index_t> get_displacements(const std::vector<index_t>& counts)
{
    // copy and do an exclusive prefix sum
    std::vector<index_t> result(counts);
    // set the total sum to zero
    index_t sum = 0;
    index_t tmp;

    // calculate the exclusive prefix sum
    typename std::vector<index_t>::iterator begin = result.begin();
    while (begin != result.end())
    {
        tmp = sum;
        // assert that the sum will still fit into the index type (MPI default:
        // int)
        MXX_ASSERT((std::size_t)sum + (std::size_t)*begin < (std::size_t) std::numeric_limits<index_t>::max());
        sum += *begin;
        *begin = tmp;
        ++begin;
    }
    return result;
}


/**
 * @brief 
 *
 * @tparam T
 * @param msgs
 * @param size
 * @param out
 * @param root
 * @param comm
 */
template <typename T>
void scatter_big(const T* msgs, size_t size, T* out, int root, const mxx::comm& comm = mxx::comm()) {
    mxx::datatype dt = mxx::get_datatype<T>().contiguous(size);
    MPI_Scatter(const_cast<T*>(msgs), 1, dt.type(), out, 1, dt.type(), root, comm);
}

/**
 * @brief Implementation of `scatterv()` for messages with elements more than MAX_INT.
 *
 * @tparam T        The type of the data.
 * @param msgs      The data to be scattered. Must be of size \f$ \sum_{i=0}^{p-1} \texttt{sizes[i]}\f$ (number of elements of type `T`).
 * @param sizes     The size (number of elements) per message per target process.
 *                  This must be a `std::vector` of size `comm.size()`.
 * @param out       Pointer to the output data. This has to point to valid
 *                  memory, which can hold at least `size` many elements of
 *                  type `T`.
 * @param recv_size The number of elements received on this process.
 * @param root      The rank of the process which scatters the data to all
 *                  other processes.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void scatterv_big(const T* msgs, const std::vector<size_t>& sizes, T* out, size_t recv_size, int root, const mxx::comm& comm = mxx::comm()) {
    // implementation of scatter for messages sizes that exceed MAX_INT
    mxx::requests reqs;
    int tag = 1234; // TODO: handle tags somewhere (as attributes in the comm?)
    if (comm.rank() == root) {
        std::size_t offset = 0;
        for (int i = 0; i < comm.size(); ++i) {
            mxx::datatype dt = mxx::get_datatype<T>().contiguous(sizes[i]);
            if (i == root) {
                // copy input into output
                std::copy(msgs+offset, msgs+offset+sizes[i], out);
            } else {
                MPI_Isend(const_cast<T*>(msgs)+offset, 1, dt.type(), i, tag, comm, &reqs.add());
            }
            offset += sizes[i];
        }
    } else {
        // create custom datatype to encapsulate the whole message
        mxx::datatype dt = mxx::get_datatype<T>().contiguous(recv_size);
        MPI_Irecv(const_cast<T*>(out), 1, dt.type(), root, tag, comm, &reqs.add());
    }
    reqs.wait();
}

/**
 * @brief   Implementation of `gather()` for large message sizes (> MAX_INT).
 *
 * @see mxx::gather
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered. Must be of size `size`.
 * @param size      The size per message. This is the number of elements which
 *                  are sent by each process. Thus `p*size` is gathered in total.
 *                  This must be the same value on all processes.
 * @param out       Pointer to the output data. On the `root` process, this has
 *                  to point to valid memory, which can hold at least `p*size`
 *                  many elements of type `T`.
 * @param root      The rank of the process which gahters the data from all
 *                  other processes to itself.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void gather_big(const T* data, size_t size, T* out, int root, const mxx::comm& comm) {
    // implementation of scatter for messages sizes that exceed MAX_INT
    mxx::datatype dt = mxx::get_datatype<T>().contiguous(size);
    MPI_Gather(const_cast<T*>(data), 1, dt.type(), out, 1, dt.type(), root, comm);
}

/**
 * @brief   Implementation for message sizes larger than MAX_INT elements.
 *
 * @see mxx::gatherv()
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered. Must be of size `size`.
 * @param size      The number of elements send from this process.
 * @param out       Pointer to the output data. On the `root` process, this has
 *                  to point to valid memory, which can hold at least
 *                  \f$ \sum_{i=0}^{p-1} \texttt{recv\_sizes[i]} \f$
 *                  many elements of type `T`.
 * @param recv_sizes    The number of elements received per process. This has
 *                      to be of size `p` on process `root`. On other processes
 *                      this can be an empty `std::vector`.
 * @param root      The rank of the process which gahters the data from all
 *                  other processes to itself.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void gatherv_big(const T* data, size_t size, T* out, const std::vector<size_t>& recv_sizes, int root, const mxx::comm& comm = mxx::comm()) {
    // implementation of scatter for messages sizes that exceed MAX_INT
    mxx::requests reqs;
    int tag = 1234; // TODO: handle tags somewhere (as attributes in the comm?)
    if (comm.rank() == root) {
        size_t offset = 0;
        for (int i = 0; i < comm.size(); ++i) {
            mxx::datatype dt = mxx::get_datatype<T>().contiguous(recv_sizes[i]);
            if (i == root) {
                // copy input into output
                std::copy(data, data+size, out+offset);
            } else {
                MPI_Irecv(const_cast<T*>(out)+offset, 1, dt.type(), i, tag, comm, &reqs.add());
            }
            offset += recv_sizes[i];
        }
    } else {
        // create custom datatype to encapsulate the whole message
        mxx::datatype dt = mxx::get_datatype<T>().contiguous(size);
        MPI_Isend(const_cast<T*>(data), 1, dt.type(), root, tag, comm, &reqs.add());
    }
    reqs.wait();
}

/**
 * @brief   Implementation of `allgather()` for large message sizes (> MAX_INT).
 *
 * @see mxx::allgather
 *
 * @tparam T        The type of the data.
 * @param data      The data to be gathered. Must be of size `size`.
 * @param size      The size per message. This is the number of elements which
 *                  are sent by each process. Thus `p*size` is gathered in total.
 *                  This must be the same value on all processes.
 * @param out       Pointer to the output data. On the `root` process, this has
 *                  to point to valid memory, which can hold at least `p*size`
 *                  many elements of type `T`.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void allgather_big(const T* data, size_t size, T* out, const mxx::comm& comm) {
    // implementation of scatter for messages sizes that exceed MAX_INT
    mxx::datatype dt = mxx::get_datatype<T>().contiguous(size);
    MPI_Allgather(const_cast<T*>(data), 1, dt.type(), out, 1, dt.type(), comm);
}

/**
 * @brief   Implementation for message sizes larger than MAX_INT elements.
 *
 * @see mxx::allgatherv()
 *
 */
template <typename T>
void allgatherv_big(const T* data, size_t size, T* out, const std::vector<size_t>& recv_sizes, const mxx::comm& comm = mxx::comm()) {
    // implementation of scatter for messages sizes that exceed MAX_INT
    mxx::requests reqs;
    int tag = 1234; // TODO: handle tags somewhere (as attributes in the comm?)
    size_t offset = 0;
    for (int i = 0; i < comm.size(); ++i) {
        // send to this rank
        mxx::datatype senddt = mxx::get_datatype<T>().contiguous(size);
        MPI_Isend(const_cast<T*>(data), 1, senddt.type(), i, tag, comm, &reqs.add());
        // receive from this rank
        mxx::datatype dt = mxx::get_datatype<T>().contiguous(recv_sizes[i]);
        MPI_Irecv(const_cast<T*>(out)+offset, 1, dt.type(), i, tag, comm, &reqs.add());
        offset += recv_sizes[i];
    }
    reqs.wait();
}

/**
 * @brief   `all2all()` implementation for message sizes larger than MAX_INT.
 *
 * @tparam T        The data type of the elements.
 * @param msgs      Pointer to memory containing the elements to be send.
 * @param size      The number of elements to send to each process. The `msgs` pointer
 *                  must contain `p*size` elements.
 * @param out       Pointer to memory for writing the received messages. Must be
 *                  able to hold at least `p*size` many elements of type `T`.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void all2all_big(const T* msgs, size_t size, T* out, const mxx::comm& comm = mxx::comm()) {
//  original version has problem when sizeof(T) * size > int_max.
//    mxx::datatype dt = mxx::get_datatype<T>().contiguous(size);
//    MPI_Alltoall(const_cast<T*>(msgs), 1, dt.type(), out, 1, dt.type(), comm);
//
//  all2all data type size limit is assumed to be max_int.
//  element count limit is max_int.
//  size is limited to max_int/P.
//  total data per rank is size * sizeof(datatype<T>) * P
//  sizeof(datatype<T>) is assume to be small relatively.
//  P is assumed to be medium in scale
//  size is assumed to be large.  so we want to try to split up size between the data type and element count.
//
//  
//  goal:  1 MPI_Alltoall to get as much data over as possible, then 1 MPI_Alltoall to take care remainders.
//  
//  
//
//  for large message size, Alltoall uses point to point anyway so need P rounds of P2P.
//  if we do size*P / max_int rounds of a2a, we end up with size*P^2 / max_int rounds of P2P
//  where as for size < max_int, we can use P rounds of P2P directly
//    and for size >= max_int, multiple rounds of a2a would size/max_int * P rounds of P2P.
//    The approach above limits it to 2x a2a so should be faster.

	if (comm.rank() == 0) printf("INFO: a2a_big called\n");

    int tag = 12346;
    // implementing this using point-to-point communication!
    mxx::requests reqs;

    mxx::datatype dt1 = mxx::get_datatype<T>();
    // get size of dt1.	
    int dt1_size = 0;
    MPI_Type_size(dt1.type(), &dt1_size);
    // get the max number of dt1 that can fit in max size data type. 
    int n_dt1_in_dtm = (mxx::max_int - 1) / dt1_size; // > 0
    mxx::datatype dtm = dt1.contiguous(n_dt1_in_dtm);
	
    // and figure out how many max size data types we have.
    size_t n_dtm = size / static_cast<size_t>(n_dt1_in_dtm); // this should be at least 1.
    size_t dt1_rem = size % static_cast<size_t>(n_dt1_in_dtm);  // this needs to be smaller than (max_int-1)/p
				// so n_dt1_in_dtm < (max_int-1)/p
				// or dt1_size > p.
				// rather than using dt1 for the last comm, just use a new custom type.
    mxx::datatype dtr = dt1.contiguous(dt1_rem);


    // for sending to self, copy directly.
    memcpy(const_cast<T*>(&(*out)) + static_cast<size_t>(comm.rank()) * size,
	const_cast<T*>(msgs) + static_cast<size_t>(comm.rank()) * size,
	size * sizeof(T));	


    // dispatch receives
    // relies on send and recv sizes matching at both ends of rank pair to ensure matching data types and number of requests.
    // again, check size against max_int.
    for (int i = 1; i < comm.size(); ++i) {
        // start with self send/recv
        int recv_from = (comm.rank() + (comm.size()-i)) % comm.size();
	size_t offset = static_cast<size_t>(recv_from) * size;

	if (size < mxx::max_int) {
	    MPI_Irecv(const_cast<T*>(&(*out)) + offset, size, dt1.type(),
                  recv_from, tag, comm, &reqs.add());
	} else {
	    // set up first part - most of data
	    if (n_dtm > 0) {
        	MPI_Irecv(const_cast<T*>(&(*out)) + offset, n_dtm, dtm.type(),
                	  recv_from, tag, comm, &reqs.add());
	    }

	    // set up second part - what doesn't fit exactly as dtm.
	    if (dt1_rem > 0) {
	       	MPI_Irecv(const_cast<T*>(&(*out)) + offset + n_dtm * static_cast<size_t>(n_dt1_in_dtm), 1, dtr.type(),
                	  recv_from, tag, comm, &reqs.add());
	    }
	}
    }
    // dispatch sends
    for (int i = 1; i < comm.size(); ++i) {
        int send_to = (comm.rank() + i) % comm.size();
	size_t offset = static_cast<size_t>(send_to) * size; 

	if (size < mxx::max_int) {
	    MPI_Isend(const_cast<T*>(msgs) + offset, size, dt1.type(),
                  send_to, tag, comm, &reqs.add());
	} else {
	
	    // set up first part - most of data
	    if (n_dtm > 0) {
        	MPI_Isend(const_cast<T*>(msgs) + offset, n_dtm, dtm.type(),
                	  send_to, tag, comm, &reqs.add());
	    }

	    // set up second part - what doesn't fit exactly as dtm.
	    if (dt1_rem > 0) {
	       	MPI_Isend(const_cast<T*>(msgs) + offset + n_dtm * static_cast<size_t>(n_dt1_in_dtm), 1, dtr.type(),
                	  send_to, tag, comm, &reqs.add());
	    }
	}

    }

    reqs.wait();

}

/**
 * @brief   Implementation for `mxx::all2allv()` for messages sizes larger than MAX_INT
 *
 * @see mxx::all2allv()
 *
 * @tparam T        The data type of the elements.
 * @param msgs      Pointer to memory containing the elements to be send.
 * @param send_sizes    A `std::vector` of size `comm.size()`, this contains
 *                      the number of elements to be send to each of the processes.
 * @param out       Pointer to memory for writing the received messages.
 * @param recv_sizes    A `std::vector` of size `comm.size()`, this contains
 *                      the number of elements to be received from each process.
 * @param comm      The communicator (`comm.hpp`). Defaults to `world`.
 */
template <typename T>
void all2allv_big(const T* msgs, const std::vector<size_t>& send_sizes, T* out, const std::vector<size_t>& recv_sizes, const mxx::comm& comm = mxx::comm()) {
    // point-to-point implementation
    // TODO: implement MPI_Alltoallw variant
    // TODO: try RMA
    MXX_ASSERT(static_cast<int>(send_sizes.size()) == comm.size());
    MXX_ASSERT(static_cast<int>(recv_sizes.size()) == comm.size());
    std::vector<size_t> send_displs = get_displacements(send_sizes);
    std::vector<size_t> recv_displs = get_displacements(recv_sizes);

    impl::all2allv_big(msgs, send_sizes, send_displs, out, recv_sizes, recv_displs, comm);
}

template <typename T>
void all2allv_big(const T* msgs, const std::vector<size_t>& send_sizes, const std::vector<size_t>& send_displs, T* out, const std::vector<size_t>& recv_sizes, const std::vector<size_t>& recv_displs, const mxx::comm& comm = mxx::comm()) {
    // point-to-point implementation
    // TODO: implement MPI_Alltoallw variant
    // TODO: try RMA
    MXX_ASSERT(static_cast<int>(send_sizes.size()) == comm.size());
    MXX_ASSERT(static_cast<int>(recv_sizes.size()) == comm.size());
 

    if (comm.rank() == 0) printf("INFO: a2aV_big called\n");

    // TODO: unify tag usage
    int tag = 12345;
    // implementing this using point-to-point communication!
    mxx::requests send_reqs;
    mxx::requests recv_reqs;

    mxx::datatype dt1 = mxx::get_datatype<T>();
    // get size of dt1.	
    int dt1_size = 0;
    MPI_Type_size(dt1.type(), &dt1_size);
    // get the max number of dt1 that can fit in max size data type. 
    int n_dt1_in_dtm = (mxx::max_int - 1) / dt1_size; // > 0
    mxx::datatype dtm = dt1.contiguous(n_dt1_in_dtm);


    // for sending to self, copy directly.
    memcpy(const_cast<T*>(&(*out)) + recv_displs[comm.rank()],
	const_cast<T*>(msgs) + send_displs[comm.rank()],
	recv_sizes[comm.rank()] * sizeof(T));	

    // dispatch receives
    // relies on send and recv sizes matching at both ends of rank pair to ensure matching data types and number of requests.
    // again, check size against max_int.
    for (int i = 1; i < comm.size(); ++i) {
        // start with self send/recv
        int recv_from = (comm.rank() + (comm.size()-i)) % comm.size();
	size_t size = recv_sizes[recv_from];
	size_t offset = recv_displs[recv_from];

	if (size < mxx::max_int) {
	    MPI_Irecv(const_cast<T*>(&(*out)) + offset, size, dt1.type(),
                  recv_from, tag, comm, &recv_reqs.add());
	} else {
	
            // and figure out how many max size data types we have.
            size_t n_dtm = size / static_cast<size_t>(n_dt1_in_dtm); // this should be at least 1.
            size_t dt1_rem = size % static_cast<size_t>(n_dt1_in_dtm);  // this needs to be smaller than (max_int-1)/p
				// so n_dt1_in_dtm < (max_int-1)/p
				// or dt1_size > p.
				// rather than using dt1 for the last comm, just use a new custom type.
			

	    // set up first part - most of data
	    if (n_dtm > 0) {
        	MPI_Irecv(const_cast<T*>(&(*out)) + offset, n_dtm, dtm.type(),
                	  recv_from, tag, comm, &recv_reqs.add());
	    }

	    // set up second part - what doesn't fit exactly as dtm.
	    if (dt1_rem > 0) {
		mxx::datatype dtr = dt1.contiguous(dt1_rem);
	       	MPI_Irecv(const_cast<T*>(&(*out)) + offset + n_dtm * static_cast<size_t>(n_dt1_in_dtm), 1, dtr.type(),
                	  recv_from, tag, comm, &recv_reqs.add());
	    }
	}
    }
    // dispatch sends
    for (int i = 1; i < comm.size(); ++i) {
        int send_to = (comm.rank() + i) % comm.size();
 	size_t size = send_sizes[send_to];
	size_t offset = send_displs[send_to];

	if (size < mxx::max_int) {
	    MPI_Isend(const_cast<T*>(msgs) + offset, size, dt1.type(),
                  send_to, tag, comm, &send_reqs.add());
	} else {
	
            // and figure out how many max size data types we have.
            size_t n_dtm = size / static_cast<size_t>(n_dt1_in_dtm); // this should be at least 1.
            size_t dt1_rem = size % static_cast<size_t>(n_dt1_in_dtm);  // this needs to be smaller than (max_int-1)/p
				// so n_dt1_in_dtm < (max_int-1)/p
				// or dt1_size > p.
				// rather than using dt1 for the last comm, just use a new custom type.
			

	    // set up first part - most of data
	    if (n_dtm > 0) {
        	MPI_Isend(const_cast<T*>(msgs) + offset, n_dtm, dtm.type(),
                	  send_to, tag, comm, &send_reqs.add());
	    }

	    // set up second part - what doesn't fit exactly as dtm.
	    if (dt1_rem > 0) {
		mxx::datatype dtr = dt1.contiguous(dt1_rem);
	       	MPI_Isend(const_cast<T*>(msgs) + offset + n_dtm * static_cast<size_t>(n_dt1_in_dtm), 1, dtr.type(),
                	  send_to, tag, comm, &send_reqs.add());
	    }
	}

    }

//	int completed = 0;
//	while (completed < send_reqs.size()) {
//		completed += send_reqs.waitsome();
//		printf("a2aV_big rank %d send waiting, completed %d\n", comm.rank(), completed);
//	}
	send_reqs.wait();

//    	completed = 0; 
//	while (completed < recv_reqs.size()) {
//		completed += recv_reqs.waitsome();
//		printf("a2aV_big rank %d recv waiting, completed %d\n", comm.rank(), completed);
//	}
    	recv_reqs.wait();

//      printf("a2aV_big rank %d done waiting\n", comm.rank());

}



} // namespace impl
} // namespace mxx

#endif // MXX_BIG_COLLECTIVE_HPP
