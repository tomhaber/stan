#ifndef STAN__ERROR_HANDLING__INVALID_ARGUMENT_VEC_HPP
#define STAN__ERROR_HANDLING__INVALID_ARGUMENT_VEC_HPP

#include <sstream>
#include <string>
#include <stan/error_handling/invalid_argument.hpp>
#include <stan/math/meta/value_type.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace error_handling {

    /**
     * Throw an invalid argument exception with a consistently formatted message.
     * 
     * This is an abstraction for all Stan functions to use when throwing
     * invalid arguments. This will allow us to change the behavior for all
     * functions at once. (We've already changed behavior mulitple times up
     * to Stan v2.5.0.)
     *
     * The message is:
     * "<function>(<typeid(value_type<T>::type)>.name()>): <name>[<i+error_index>] <msg1><y>"
     *    where error_index is the value of stan::error_index::value 
     * which indicates whether the message should be 0 or 1 indexed.
     *
     * @tparam T Type of variable
     * @param function Name of the function
     * @param name Name of the variable
     * @param y Variable
     * @param i Index
     * @param msg1 Message to print before the variable
     * @param msg2 Message to print after the variable
     * @throw std::invalid_argument
     */
    template <typename T>
    inline void invalid_argument_vec(const std::string& function,
                                 const std::string& name,
                                 const T& y,
                                 const size_t i,
                                 const std::string& msg1,
                                 const std::string& msg2) {
      std::ostringstream vec_name_stream;
      vec_name_stream << name
                      << "[" << stan::error_index::value + i << "]";
      std::string vec_name(vec_name_stream.str());
      invalid_argument(function, vec_name.c_str(), stan::get(y, i) , msg1, msg2);
    }

    /**
     * Throw an invalid argument exception with a consistently formatted message.
     * 
     * This is an abstraction for all Stan functions to use when throwing
     * invalid arguments. This will allow us to change the behavior for all
     * functions at once. (We've already changed behavior mulitple times up
     * to Stan v2.5.0.)
     *
     * The message is:
     * "<function>(<typeid(T)>.name()>): <name>[<i+error_index>] <msg1><y>"
     *   where error_index is the value of stan::error_index::value 
     * which indicates whether the message should be 0 or 1 indexed.
     *
     * @tparam T Type of variable
     * @param function Name of the function
     * @param name Name of the variable
     * @param y Variable
     * @param i Index
     * @param msg Message to print before the variable
     * @throw std::invalid_argument
     */
    template <typename T>
    inline void invalid_argument_vec(const std::string& function,
                                 const std::string& name,
                                 const T& y,
                                 const size_t i,
                                 const std::string& msg) {
      invalid_argument_vec(function, name, y, i, msg, "");
    }
    
  }
}
#endif
