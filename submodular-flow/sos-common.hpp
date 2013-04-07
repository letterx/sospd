#ifndef _SOS_COMMON_HPP_
#define _SOS_COMMON_HPP_

#include <stdint.h>
#include <limits>
#include <vector>
#include <memory>
#include <assert.h>
#include <stdexcept>
#include <string>

#ifndef DNO_ASSERT
#define ASSERT(cond) do { if (!(cond)) { throw std::logic_error((std::string("Assertion failure at " __FILE__ ":")+std::to_string(__LINE__)+std::string(" -- " #cond)).c_str() ); }} while(0)
#else
#define ASSERT(cond) (void)0
#endif

#endif
