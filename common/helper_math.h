#ifndef VECTOR_MATH_H
#define VECTOR_MATH_H

#include "cuda_runtime.h"

////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef __CUDACC__
#include <cmath>

inline float fminf(float a, float b)
{
  return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
  return a < b ? a : b;
}

inline int max(int a, int b)
{
  return a > b ? a : b;
}

inline int min(int a, int b)
{
  return a < b ? a : b;
}

inline float rsqrtf(float x)
{
	return 1.0f / sqrtf(x);
}
#endif

// float functions
////////////////////////////////////////////////////////////////////////////////

// lerp
inline __device__ __host__ float lerp(const float &a, const float &b, const float &t)
{
	return a + t*(b-a);
}

// clamp
inline __device__ __host__ float clamp(const float &f, const float &a, const float &b)
{
	return fmaxf(a, fminf(f, b));
}

// int2 functions
////////////////////////////////////////////////////////////////////////////////

// negate
inline __host__ __device__ int2 operator-(const int2 &a)
{
	return make_int2(-a.x, -a.y);
}

// addition
inline __host__ __device__ int2 operator+(const int2 &a, const int2 &b)
{
	return make_int2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(int2 &a, const int2 &b)
{
	a.x += b.x; a.y += b.y;
}

// subtract
inline __host__ __device__ int2 operator-(const int2 &a, const int2 &b)
{
	return make_int2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(int2 &a, const int2 &b)
{
	a.x -= b.x; a.y -= b.y;
}

// multiply
inline __host__ __device__ int2 operator*(const int2 &a, const int2 &b)
{
	return make_int2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ int2 operator*(const int2 &a, const int &s)
{
	return make_int2(a.x * s, a.y * s);
}
inline __host__ __device__ int2 operator*(const int &s, const int2 &a)
{
	return make_int2(a.x * s, a.y * s);
}
inline __host__ __device__ void operator*=(int2 &a, const int &s)
{
	a.x *= s; a.y *= s;
}

// float2 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ float2 make_float2(const float &s)
{
	return make_float2(s, s);
}
inline __host__ __device__ float2 make_float2(const int2 &a)
{
	return make_float2(float(a.x), float(a.y));
}

// negate
inline __host__ __device__ float2 operator-(const float2 &a)
{
	return make_float2(-a.x, -a.y);
}

// addition
inline __host__ __device__ float2 operator+(const float2 &a, const float2 &b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, const float2 &b)
{
	a.x += b.x; a.y += b.y;
}

// subtract
inline __host__ __device__ float2 operator-(const float2 &a, const float2 &b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, const float2 &b)
{
	a.x -= b.x; a.y -= b.y;
}

// multiply
inline __host__ __device__ float2 operator*(const float2 &a, const float2 &b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ float2 operator*(const float2 &a, const float &s)
{
	return make_float2(a.x * s, a.y * s);
}
inline __host__ __device__ float2 operator*(const float &s, const float2 &a)
{
	return make_float2(a.x * s, a.y * s);
}
inline __host__ __device__ void operator*=(float2 &a, const float &s)
{
	a.x *= s; a.y *= s;
}

// divide
inline __host__ __device__ float2 operator/(const float2 &a, const float2 &b)
{
	return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ float2 operator/(const float2 &a, const float &s)
{
	float inv = 1.0f / s;
	return a * inv;
}
inline __host__ __device__ float2 operator/(const float &s, const float2 &a)
{
	float inv = 1.0f / s;
	return a * inv;
}
inline __host__ __device__ void operator/=(float2 &a, const float &s)
{
	float inv = 1.0f / s;
	a *= inv;
}

// lerp
inline __device__ __host__ float2 lerp(const float2 &a, const float2 &b, const float &t)
{
	return a + t*(b-a);
}

// clamp
inline __device__ __host__ float2 clamp(const float2 &v, const float &a, const float &b)
{
	return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}

inline __device__ __host__ float2 clamp(const float2 &v, const float2 &a, const float2 &b)
{
	return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}

// dot product
inline __host__ __device__ float dot(const float2 &a, const float2 &b)
{
	return a.x * b.x + a.y * b.y;
}

// squared length
inline __host__ __device__ float sqlength(const float2 &v)
{
	return dot(v, v);
}

// length
inline __host__ __device__ float length(const float2 &v)
{
	return sqrtf(sqlength(v));
}

// normalize
inline __host__ __device__ float2 normalize(const float2 &v)
{
	float invLen = rsqrtf(sqlength(v));
	return v * invLen;
}

// floor
inline __host__ __device__ float2 floor(const float2 &v)
{
	return make_float2(floor(v.x), floor(v.y));
}

// reflect
inline __host__ __device__ float2 reflect(const float2 &i, const float2 &n)
{
	return i - 2.0f * n * dot(n,i);
}

// absolute value
inline __host__ __device__ float2 fabs(const float2 &v)
{
	return make_float2(fabs(v.x), fabs(v.y));
}

// float3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ float3 make_float3(const float &s)
{
	return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(const float2 &a)
{
	return make_float3(a.x, a.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(const float2 &a, const float &s)
{
	return make_float3(a.x, a.y, s);
}
inline __host__ __device__ float3 make_float3(const float4 &a)
{
	return make_float3(a.x, a.y, a.z);  // discards w
}
inline __host__ __device__ float3 make_float3(const int3 &a)
{
	return make_float3(float(a.x), float(a.y), float(a.z));
}

// negate
inline __host__ __device__ float3 operator-(const float3 &a)
{
	return make_float3(-a.x, -a.y, -a.z);
}

// min
static __inline__ __host__ __device__ float3 fminf(const float3 &a, const float3 &b)
{
	return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}

// max
static __inline__ __host__ __device__ float3 fmaxf(const float3 &a, const float3 &b)
{
	return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}

// addition
inline __host__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, const float3 &b)
{
	a.x += b.x; a.y += b.y; a.z += b.z;
}

// subtract
inline __host__ __device__ float3 operator-(const float3 &a, const float3 &b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float3 operator-(const float3 &a, const float &b)
{
	return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ void operator-=(float3 &a, const float3 &b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
inline __host__ __device__ float3 operator*(const float3 &a, const float3 &b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ float3 operator*(const int3 &a, const float3 &b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ float3 operator*(const float3 &a, const float &s)
{
	return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ float3 operator*(const float &s, const float3 &a)
{
	return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ void operator*=(float3 &a, const float &s)
{
	a.x *= s; a.y *= s; a.z *= s;
}

// divide
inline __host__ __device__ float3 operator/(const float3 &a, const float3 &b)
{
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ float3 operator/(const float3 &a, const float &s)
{
	float inv = 1.0f / s;
	return a * inv;
}
inline __host__ __device__ float3 operator/(const float &s, const float3 &a)
{
	float inv = 1.0f / s;
	return a * inv;
}
inline __host__ __device__ void operator/=(float3 &a, const float &s)
{
	float inv = 1.0f / s;
	a *= inv;
}

// lerp
inline __device__ __host__ float3 lerp(const float3 &a, const float3 &b, const float &t)
{
	return a + t*(b-a);
}

// clamp
inline __device__ __host__ float3 clamp(const float3 &v, const float &a, const float &b)
{
	return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __device__ __host__ float3 clamp(const float3 &v, const float3 &a, const float3 &b)
{
	return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

// dot product
inline __host__ __device__ float dot(const float3 &a, const float3 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

// cross product
inline __host__ __device__ float3 cross(const float3 &a, const float3 &b)
{
	return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

// squared length
inline __host__ __device__ float sqlength(const float3 &v)
{
	return dot(v, v);
}

// length
inline __host__ __device__ float length(const float3 &v)
{
	return sqrtf(sqlength(v));
}

// normalize
inline __host__ __device__ float3 normalize(const float3 &v)
{
	float invLen = rsqrtf(sqlength(v));
	return v * invLen;
}

// floor
inline __host__ __device__ float3 floor(const float3 &v)
{
	return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

// reflect
inline __host__ __device__ float3 reflect(const float3 &i, const float3 &n)
{
	return i - 2.0f * n * dot(n,i);
}

// absolute value
inline __host__ __device__ float3 fabs(const float3 &v)
{
	return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}

inline __host__ __device__ float3 rotate(const float3 &v, const float3 &ort, const float &angle)
{
	float vnorm = length(ort);
	float ct = cosf(angle);
	float st = sinf(angle);
	float x = ort.x, y = ort.y, z = ort.z;

	float a11, a12, a13, a21, a22, a23, a31, a32, a33;

	a11 = x*x + (y*y + z*z)*ct;
	a11 /= vnorm*vnorm;

	a22 = y*y + (x*x + z*z)*ct;
	a22 /= vnorm*vnorm;

	a33 = z*z + (x*x + y*y)*ct;
	a33 /= vnorm*vnorm;

	a12 = x*y*(1-ct)-z*vnorm*st;
	a12 /= vnorm*vnorm;

	a21 = x*y*(1-ct)+z*vnorm*st;
	a21 /= vnorm*vnorm;

	a13 = x*z*(1-ct)+y*vnorm*st;
	a13 /= vnorm*vnorm;

	a31 = x*z*(1-ct)-y*vnorm*st;
	a31 /= vnorm*vnorm;

	a23 = y*z*(1-ct)-x*vnorm*st;
	a23 /= vnorm*vnorm;

	a32 = y*z*(1-ct)+x*vnorm*st;
	a32 /= vnorm*vnorm;

	return make_float3(
			a11*v.x+a12*v.y+a13*v.z,
			a21*v.x+a22*v.y+a23*v.z,
			a31*v.x+a32*v.y+a33*v.z);
}

// float4 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ float4 make_float4(const float &s)
{
	return make_float4(s, s, s, s);
}
inline __host__ __device__ float4 make_float4(const float3 &a)
{
	return make_float4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ float4 make_float4(const float3 &a, const float &w)
{
	return make_float4(a.x, a.y, a.z, w);
}
inline __host__ __device__ float4 make_float4(const float2 &a, const float2 &b)
{
	return make_float4(a.x, a.y, b.x, b.y);
}
inline __host__ __device__ float4 make_float4(const int4 &a)
{
	return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

// show a float4 as a float3
inline __host__ __device__ float3& as_float3(const float4 &v)
{
	return *(float3*)&v;
}

// negate
inline __host__ __device__ float4 operator-(const float4 &a)
{
	return make_float4(-a.x, -a.y, -a.z, -a.w);
}

// min
static __inline__ __host__ __device__ float4 fminf(const float4 &a, const float4 &b)
{
	return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}

// max
static __inline__ __host__ __device__ float4 fmaxf(const float4 &a, const float4 &b)
{
	return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

// addition
inline __host__ __device__ float4 operator+(const float4 &a, const float4 &b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, const float4 &b)
{
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

// subtract
inline __host__ __device__ float4 operator-(const float4 &a, const float4 &b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(float4 &a, const float4 &b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

// multiply
inline __host__ __device__ float4 operator*(const float4 &a, const float &s)
{
	return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline __host__ __device__ float4 operator*(const float &s, const float4 &a)
{
	return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline __host__ __device__ void operator*=(float4 &a, const float &s)
{
	a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}

// divide
inline __host__ __device__ float4 operator/(const float4 &a, const float4 &b)
{
	return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ float4 operator/(const float4 &a, const float &s)
{
	float inv = 1.0f / s;
	return a * inv;
}
inline __host__ __device__ float4 operator/(const float &s, const float4 &a)
{
	float inv = 1.0f / s;
	return a * inv;
}
inline __host__ __device__ void operator/=(float4 &a, float s)
{
	float inv = 1.0f / s;
	a *= inv;
}

// lerp
inline __device__ __host__ float4 lerp(const float4 &a, const float4 &b, const float &t)
{
	return a + t*(b-a);
}

// clamp
inline __device__ __host__ float4 clamp(const float4 &v, const float &a, const float &b)
{
	return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

inline __device__ __host__ float4 clamp(const float4 &v, const float4 &a, const float4 &b)
{
	return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

// dot product
inline __host__ __device__ float dot(const float4 &a, const float4 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// squared length
inline __host__ __device__ float sqlength(const float4 &v)
{
	return dot(v, v);
}

// length
inline __host__ __device__ float length(const float4 &v)
{
	return sqrtf(sqlength(v));
}

// normalize
inline __host__ __device__ float4 normalize(const float4 &v)
{
	float invLen = rsqrtf(sqlength(v));
	return v * invLen;
}

// floor
inline __host__ __device__ float4 floor(const float4 &v)
{
	return make_float4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}

// absolute value
inline __host__ __device__ float4 fabs(const float4 &v)
{
	return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

// int3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ int3 make_int3(const int &s)
{
	return make_int3(s, s, s);
}
inline __host__ __device__ int3 make_int3(const float3 &a)
{
	return make_int3(int(a.x), int(a.y), int(a.z));
}

// negate
inline __host__ __device__ int3 operator-(const int3 &a)
{
	return make_int3(-a.x, -a.y, -a.z);
}

// min
inline __host__ __device__ int3 min(const int3 &a, const int3 &b)
{
	return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

// max
inline __host__ __device__ int3 max(const int3 &a, const int3 &b)
{
	return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

// addition
inline __host__ __device__ int3 operator+(const int3 &a, const int3 &b)
{
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(int3 &a, const int3 &b)
{
	a.x += b.x; a.y += b.y; a.z += b.z;
}

// subtract
inline __host__ __device__ int3 operator-(const int3 &a, const int3 &b)
{
	return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ void operator-=(int3 &a, const int3 &b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
inline __host__ __device__ int3 operator*(const int3 &a, const int3 &b)
{
	return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ int3 operator*(const int3 &a, const int &s)
{
	return make_int3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ int3 operator*(const int &s, const int3 &a)
{
	return make_int3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ void operator*=(int3 &a, const int &s)
{
	a.x *= s; a.y *= s; a.z *= s;
}

// divide
inline __host__ __device__ int3 operator/(const int3 &a, const int3 &b)
{
	return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ int3 operator/(const int3 &a, const int &s)
{
	return make_int3(a.x / s, a.y / s, a.z / s);
}
inline __host__ __device__ int3 operator/(const int &s, const int3 &a)
{
	return make_int3(a.x / s, a.y / s, a.z / s);
}
inline __host__ __device__ void operator/=(int3 &a, const int &s)
{
	a.x /= s; a.y /= s; a.z /= s;
}

// clamp
inline __device__ __host__ int clamp(const int &f, const int &a, const int &b)
{
	return max(a, min(f, b));
}

inline __device__ __host__ int3 clamp(const int3 &v, const int &a, const int &b)
{
	return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __device__ __host__ int3 clamp(const int3 &v, const int3 &a, const int3 &b)
{
	return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}


// uint3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ uint3 make_uint3(const uint &s)
{
	return make_uint3(s, s, s);
}
inline __host__ __device__ uint3 make_uint3(const float3 &a)
{
	return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

// min
inline __host__ __device__ uint3 min(const uint3 &a, const uint3 &b)
{
	return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

// max
inline __host__ __device__ uint3 max(const uint3 &a, const uint3 &b)
{
	return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

// addition
inline __host__ __device__ uint3 operator+(const uint3 &a, const uint3 &b)
{
	return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(uint3 &a, const uint3 &b)
{
	a.x += b.x; a.y += b.y; a.z += b.z;
}

// subtract
inline __host__ __device__ uint3 operator-(const uint3 &a, const uint3 &b)
{
	return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ void operator-=(uint3 &a, const uint3 &b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
inline __host__ __device__ uint3 operator*(const uint3 &a, const uint3 &b)
{
	return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ uint3 operator*(const uint3 &a, const uint &s)
{
	return make_uint3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ uint3 operator*(const uint &s, const uint3 &a)
{
	return make_uint3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ void operator*=(uint3 &a, const uint &s)
{
	a.x *= s; a.y *= s; a.z *= s;
}

// divide
inline __host__ __device__ uint3 operator/(const uint3 &a, const uint3 &b)
{
	return make_uint3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ uint3 operator/(const uint3 &a, const uint &s)
{
	return make_uint3(a.x / s, a.y / s, a.z / s);
}
inline __host__ __device__ uint3 operator/(const uint &s, const uint3 &a)
{
	return make_uint3(a.x / s, a.y / s, a.z / s);
}
inline __host__ __device__ void operator/=(uint3 &a, const uint &s)
{
	a.x /= s; a.y /= s; a.z /= s;
}

// clamp
inline __device__ __host__ uint clamp(const uint &f, const uint &a, const uint &b)
{
	return max(a, min(f, b));
}

inline __device__ __host__ uint3 clamp(const uint3 &v, const uint &a, const uint &b)
{
	return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __device__ __host__ uint3 clamp(const uint3 &v, const uint3 &a, const uint3 &b)
{
	return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

#endif
