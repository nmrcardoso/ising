
#ifndef ALLOC_H
#define ALLOC_H


#include <cstdlib>
#include <enum.h>


namespace ising {


  void printPeakMemUsage();
  void assertAllMemFree();
  void FreeAllMemory();

  /**
     @return peak device memory allocated
   */
  long dev_allocated_peak();

  /**
     @return peak pinned memory allocated
   */
  long pinned_allocated_peak();

  /**
     @return peak mapped memory allocated
   */
  long mapped_allocated_peak();

  /**
     @return peak host memory allocated
   */
  long host_allocated_peak();
  
  
  /**
     @return are we using managed memory for device allocations
  */
  bool use_managed_memory();

  /**
     @return is prefetching support enabled (assumes managed memory is enabled)
  */
  bool is_prefetch_enabled();

  /*
   * The following functions should not be called directly.  Use the
   * macros below instead.
   */
  void *dev_malloc_(const char *func, const char *file, int line, size_t size);
  void *dev_pinned_malloc_(const char *func, const char *file, int line, size_t size);
  void *safe_malloc_(const char *func, const char *file, int line, size_t size);
  void *pinned_malloc_(const char *func, const char *file, int line, size_t size);
  void *mapped_malloc_(const char *func, const char *file, int line, size_t size);
  void dev_free_(const char *func, const char *file, int line, void *ptr);
  void dev_pinned_free_(const char *func, const char *file, int line, void *ptr);
  void host_free_(const char *func, const char *file, int line, void *ptr);
  void *managed_malloc_(const char *func, const char *file, int line, size_t size);
  void managed_free_(const char *func, const char *file, int line, void *ptr);
  
    /*
    @brief Get device view of a host-mapped pointer
   */
  void *get_mapped_device_pointer_(const char *func, const char *file, int line, const void *ptr);



  // strip path from __FILE__
  inline constexpr const char* str_end(const char *str) { return *str ? str_end(str + 1) : str; }
  inline constexpr bool str_slant(const char *str) { return *str == '/' ? true : (*str ? str_slant(str + 1) : false); }
  inline constexpr const char* r_slant(const char* str) { return *str == '/' ? (str + 1) : r_slant(str - 1); }
  inline constexpr const char* file_name(const char* str) { return str_slant(str) ? r_slant(str_end(str)) : str; }

  FieldLocation get_pointer_location(const void *ptr);

} // namespace alloc

#define dev_malloc(size) ising::dev_malloc_(__func__, ising::file_name(__FILE__), __LINE__, size)
#define dev_pinned_malloc(size) ising::dev_pinned_malloc_(__func__, ising::file_name(__FILE__), __LINE__, size)
#define safe_malloc(size) ising::safe_malloc_(__func__, ising::file_name(__FILE__), __LINE__, size)
#define pinned_malloc(size) ising::pinned_malloc_(__func__, ising::file_name(__FILE__), __LINE__, size)
#define mapped_malloc(size) ising::mapped_malloc_(__func__, ising::file_name(__FILE__), __LINE__, size)
#define dev_free(ptr) ising::dev_free_(__func__, ising::file_name(__FILE__), __LINE__, ptr)
#define dev_pinned_free(ptr) ising::dev_pinned_free_(__func__, ising::file_name(__FILE__), __LINE__, ptr)
#define host_free(ptr) ising::host_free_(__func__, ising::file_name(__FILE__), __LINE__, ptr)
#define managed_malloc(size) ising::managed_malloc_(__func__, ising::file_name(__FILE__), __LINE__, size)
#define managed_free(ptr) ising::managed_free_(__func__, ising::file_name(__FILE__), __LINE__, ptr)
#define get_mapped_device_pointer(ptr) ising::get_mapped_device_pointer_(__func__, ising::file_name(__FILE__), __LINE__, ptr)







#endif

