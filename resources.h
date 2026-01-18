//
//  resources.h
//  Codeforces
//
//  Created by Mauryavardhan Singh on 20/10/24.
//

#ifndef resources_h
#define resources_h


#endif /* resources_h */


// __builtin_ffs(x)
// This function returns 1 + least significant 1-bit of x. If x == 0, returns 0. Here x is int, this function with suffix 'l' gets a long argument and with suffix 'll' gets a long long argument.

// e.g. __builtin_ffs(10) = 2 because 10 is '...10 1 0' in base 2 and first 1-bit from right is at index 1 (0-based) and function returns 1 + index.

// three)

// __builtin_clz(x)
// This function returns number of leading 0-bits of x which starts from most significant bit position. x is unsigned int and like previous function this function with suffix 'l gets a unsigned long argument and with suffix 'll' gets a unsigned long long argument. If x == 0, returns an undefined value.

// e.g. __builtin_clz(16) = 27 because 16 is ' ... 10000'. Number of bits in a unsigned int is 32. so function returns 32 — 5 = 27.

// four)

// __builtin_ctz(x)
// This function returns number of trailing 0-bits of x which starts from least significant bit position. x is unsigned int and like previous function this function with suffix 'l' gets a unsigned long argument and with suffix 'll' gets a unsigned long long argument. If x == 0, returns an undefined value.

// e.g. __builtin_ctz(16) = 4 because 16 is '...1 0000 '. Number of trailing 0-bits is 4.

// five)

// __builtin_popcount(x)
// This function returns number of 1-bits of x. x is unsigned int and like previous function this function with suffix 'l' gets a unsigned long argument and with suffix 'll' gets a unsigned long long argument. If x == 0, returns an undefined value.

// e.g. __builtin_popcount(14) = 3 because 14 is '... 111 0' and has three 1-bits.



//For Generating Random numbers

#include <random>
mt19937 mt(727);
uniform_int_distribution uni(1,3);

typedef long long ll;
typedef struct SegmentTree{
    ll n;
    vector<ll> segments;
    void init(ll size){
        n=size;
        segments.resize(4*n);
    }
    ll combine(ll left,ll right){
        return left + right;
    }
    void build(vector<ll> &a,ll v, ll tl, ll tr) {//v is 1-based vertex while rest are 0-based ranges
        if (tl == tr) {
            segments[v] =a[tl];
        } else {
            ll tm = (tl + tr) / 2;
            build(a,v*2, tl, tm);
            build(a,v*2+1, tm+1, tr);
            segments[v] = combine(segments[v*2],segments[v*2+1]);
        }
    }
    void build(ll size,vector<ll> a){
        init(size);
        build(a,1,0,n-1);
    }
    ll get(ll v, ll tl, ll tr, ll l, ll r) {
        if (l > r)
            return 0;// return default value acc to combine func e.g. INF for combine -> min
        if (l == tl && r == tr) {
            return segments[v];
        }
        ll tm = (tl + tr) / 2;
        return combine(get(v*2, tl, tm, l, min(r, tm)),get(v*2+1, tm+1, tr, max(l, tm+1), r));
    }
    ll get(ll l,ll r){
        return get(1,0,n-1,l,r);
    }
    void update(ll v, ll tl, ll tr, ll pos, ll new_val) {
        if (tl == tr) {
            segments[v] = new_val;
        } else {
            ll tm = (tl + tr) / 2;
            if (pos <= tm)
                update(v*2, tl, tm, pos, new_val);
            else
                update(v*2+1, tm+1, tr, pos, new_val);
            segments[v] = combine(segments[v*2],segments[v*2+1]);
        }
    }
    void update(ll pos,ll new_val){
        update(1,0,n-1,pos,new_val);
    }
}SegmentTree;


// Lazy Propagation of Segment Tree

struct SegmentTree {
    ll n;
    vector<ll> segments;
    vector<ll> lazy;

    // initialize segment tree and lazy arrays
    void init(ll size) {
        n = size;
        segments.assign(4*n, 0);
        lazy.assign(4*n, 0);
    }

    // combining two child values
    ll combine(ll left, ll right) {
        return left + right;
    }

    // build from initial array a
    void build(vector<ll> &a, ll v, ll tl, ll tr) {
        if (tl == tr) {
            segments[v] = a[tl];
        } else {
            ll tm = (tl + tr) / 2;
            build(a, v * 2, tl, tm);
            build(a, v * 2 + 1, tm + 1, tr);
            segments[v] = combine(segments[v * 2], segments[v * 2 + 1]);
        }
    }

    void build(ll size, vector<ll> a) {
        init(size);
        build(a, 1, 0, n - 1);
    }

    // propagate pending updates to children
    void push(ll v, ll tl, ll tr) {
        if (lazy[v] != 0) {
            ll add = lazy[v];
            ll tm = (tl + tr) / 2;
            // update left child
            segments[v * 2] += add * (tm - tl + 1);
            lazy[v * 2] += add;
            // update right child
            segments[v * 2 + 1] += add * (tr - tm);
            lazy[v * 2 + 1] += add;
            // clear current
            lazy[v] = 0;
        }
    }

    // range add: add 'addend' to each element in [l, r]
    void update_range(ll v, ll tl, ll tr, ll l, ll r, ll addend) {
        if (l > r)
            return;
        if (l == tl && r == tr) {
            segments[v] += addend * (tr - tl + 1);
            lazy[v] += addend;
        } else {
            push(v, tl, tr);
            ll tm = (tl + tr) / 2;
            update_range(v * 2, tl, tm, l, min(r, tm), addend);
            update_range(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r, addend);
            segments[v] = combine(segments[v * 2], segments[v * 2 + 1]);
        }
    }

    // wrapper for range update
    void update_range(ll l, ll r, ll addend) {
        update_range(1, 0, n - 1, l, r, addend);
    }

    // point update (set value)
    void update_point(ll v, ll tl, ll tr, ll pos, ll new_val) {
        if (tl == tr) {
            segments[v] = new_val;
        } else {
            push(v, tl, tr);
            ll tm = (tl + tr) / 2;
            if (pos <= tm)
                update_point(v * 2, tl, tm, pos, new_val);
            else
                update_point(v * 2 + 1, tm + 1, tr, pos, new_val);
            segments[v] = combine(segments[v * 2], segments[v * 2 + 1]);
        }
    }

    void update(ll pos, ll new_val) {
        update_point(1, 0, n - 1, pos, new_val);
    }

    // range query sum
    ll get(ll v, ll tl, ll tr, ll l, ll r) {
        if (l > r)
            return 0;
        if (l == tl && r == tr) {
            return segments[v];
        }
        push(v, tl, tr);
        ll tm = (tl + tr) / 2;
        return combine(
            get(v * 2, tl, tm, l, min(r, tm)),
            get(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r)
        );
    }

    ll get(ll l, ll r) {
        return get(1, 0, n - 1, l, r);
    }
};

// Usage example:
// vector<ll> a = {1,2,3,4,5};
// SegmentTree st;
// st.build(a.size(), a);
// st.update_range(1, 3, 10); // add 10 to a[1..3]
// cout << st.get(0, 4) << endl;


typedef struct DSU{//never use parents array directly use find()
    vector<ll> parents;
    vector<ll> sizes;
    ll components;
    ll n;
    void init(ll sizeOfDSU){
        n=sizeOfDSU;
        components=n;
        for(ll i=0;i<n;i++){
            parents.push_back(i);
            sizes.push_back(1);
        }
    }
    ll find(ll x) {
        return parents[x] == x ? x : (parents[x] = find(parents[x]));
    }
    void unite(ll x, ll y) {
        ll x_root = find(x);
        ll y_root = find(y);
        if (x_root == y_root) { return; }
 
        if (sizes[x_root] < sizes[y_root]) { swap(x_root, y_root); }
        sizes[x_root] += sizes[y_root];
        parents[y_root] = x_root;
        components--;
    }
    bool connected(ll x, ll y) { return find(x) == find(y); }
    
    ll sizeOf(ll x) {
            return sizes[find(x)];
    }
}DSU;
//DSU dsu;
//dsu.init(5);
bool isPrime(int n) {
    if (n < 2) return false;
    for (int i = 2; i * i <= n; ++i)
        if (n % i == 0) return false;
    return true;
}

long long binpow(long long a, long long b, long long m=mod) {
    a %= m;
    long long res = 1;
    while (b > 0) {
        if (b & 1){
            res = res * a % m;
            res%=m;
        }
        a = a * a % m;
        b >>= 1;
    }
    return res%m;
}
ll dp[100001];

long long mod_inverse(long long n,long long m=mod){
    long long p=m-2;
    //using binpow
    n %= m;
    long long res = 1;
    while (p > 0) {
        if (p & 1){
            res = res * n % m;
            res%=m;
        }
        n = n * n % m;
        p >>= 1;
    }
    return res%m;
}

ll ncr(ll n, ll r){
    return ((dp[n]%mod)*mod_inverse(dp[r])%mod*mod_inverse(dp[n-r])%mod)%mod;
}

// Ordered Multiset Data Structure

#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace std;
using namespace __gnu_pbds;

// Typedef for convenience
typedef long long ll;

// Ordered multiset using (value, unique id) pairs
typedef pair<ll, ll> pll;

template <typename T>
using ordered_multiset = tree<pll, null_type, less<pll>, rb_tree_tag, tree_order_statistics_node_update>;

// Helper structure to wrap all operations cleanly
struct OrderedMultiset {
    ordered_multiset<ll> s;
    ll id = 0;  // unique id for each element to handle duplicates

    // Insert element
    void insert(ll x) {
        s.insert({x, id++});
    }

    // Erase one occurrence of element x
    void erase(ll x) {
        auto it = s.lower_bound({x, -1});  // first occurrence
        if (it != s.end() && it->first == x) {
            s.erase(it);
        }
    }

    // Find position (rank) of first occurrence of x (0-based index)
    ll position(ll x) {
        return s.order_of_key({x, -1});
    }

    // Find k-th smallest element (0-based)
    ll at(ll index) {
        return s.find_by_order(index)->first;
    }

    // Current size of the multiset (total elements including duplicates)
    ll size() {
        return s.size();
    }
    // Clear the entire multiset
    void clear() {
        s.clear();
        id = 0;
    }
};


//Basic Structure of a Tree
const ll N=1e7;
bool vis[N];
vvl g(N);

void dfs(ll vertex){
    /* Take action on vertex
     after entering the vertex*/
    vis[vertex]=true;
    for (ll child : g[vertex]){
        /*Take action on child node
         before entering the child node*/
        if (vis[child]) continue;
        dfs(child);
        /*Take action on child node
         after entering the child node*/
    }
    /*Take action on vertex
     before exiting the vertex*/
}


//Binary Lifting + Depth of Each node
//vvl memo(n+1,vl(27,-1));
//vl depth(n+1,0);

void bin_lift(ll vertex, ll par, vvl &memo, ll n, vl &depth, ll lev){
    for (ll i=1; i<=25; i++){
        if (memo[vertex][i-1]==-1) break;
        memo[vertex][i]=memo[memo[vertex][i-1]][i-1];
    }
    for (ll child : g[vertex]){
        if (child!=par){
            memo[child][0]=vertex;
            lev++;
            depth[child]=lev;
            bin_lift(child,vertex,memo,n,depth,lev);
            lev--;
        }
    }
}

//Lowest Common Ancestor for a Tree

ll LCA(ll a1, ll b1, vl &depth, vvl &memo){ //Assuming a1>b1
    ll rem=depth[a1]-depth[b1];
    for (ll j=0; j<=25; j++){
        if (rem&(1<<j)){
            a1=memo[a1][j];
            if (a1==-1){
                break;
            }
        }
    }
    if (a1==b1){
        return a1;
    }
    for (ll j=25; j>=0; j--){
        if (memo[a1][j]!=memo[b1][j]){
            a1=memo[a1][j];
            b1=memo[b1][j];
        }
    }
    return memo[a1][0];
}

//Merge Sort along with code for counting inversions

ll merge(vl &arr, ll left, ll mid, ll right){
    ll invCount=0;
    vl l,r;
    for (ll i=left; i<=mid; i++){
        l.PB(arr[i]);
    }
    for (ll i=mid+1; i<=right; i++){
        r.PB(arr[i]);
    }
    
    ll i=0,j=0,k=left;
    while(i<l.size() && j<r.size()){
        if (l[i]<=r[j]){
            arr[k]=l[i];
            i++;
        }
        else{
            arr[k]=r[j];
            invCount+=l.size()-i;
            j++;
        }
        k++;
    }
    while(i<l.size()){
        arr[k]=l[i];
        i++;
        k++;
    }
    while(j<r.size()){
        arr[k]=r[j];
        j++;
        k++;
    }
    return invCount;
}

ll mergeSort(vl &arr, ll left, ll right){
    ll invCount=0;
    if (left>=right){
        return 0;
    }
    ll mid=left+(right-left)/2;
    invCount+=mergeSort(arr,left,mid);
    invCount+=mergeSort(arr, mid+1, right);
    invCount+=merge(arr,left,mid,right);
    return invCount;
}

// Quick Sort Algorithm

ll pid(vl &arr, ll low, ll high){
    ll i=low;
    ll j=high;
    while(i<j){
        while(arr[i]<=arr[low] and i<high) i++;
        while(arr[j]>arr[low] and j>low) j--;
        if (i<j) swap(arr[i],arr[j]);
    }
    swap(arr[j],arr[low]);
    return j;
}

void quickSort(vl &arr, ll low, ll high){
    if (low>=high) return;
    ll p=pid(arr,low,high);
    quickSort(arr,low,p-1);
    quickSort(arr,p+1,high);
}

string toBinary(long long x) {
    if (x == 0) return "0";
    string s;
    while (x > 0) {
        s.push_back('0' + (x & 1)); 
        x >>= 1;                    
    }
    reverse(s.begin(), s.end());
    return s;
}
void print(vector<ll> &v){
    for(auto &x:v){
        cout<<x<<" ";
    }
    cout<<"\n";
}

ll gcd(ll a, ll b){
    if(b == 0) return a;
    return gcd(b, a % b);
}

ll power(ll base, ll exp){
    ll result = 1;
    while(exp > 0){
        if(exp % 2 == 1){
            result *= base;
        }
        base *= base;
        exp /= 2;
    }
    return result;
}

const ll N = 201;
vector<bool> is_prime(N, true);
vector<ll> primes;

void sieve() {
    is_prime[0] = is_prime[1] = false;
    for (ll i = 2; i < N; i++) {
        if (is_prime[i]) {
            primes.push_back(i);
            for (ll j = i * i; j < N; j += i) {
                is_prime[j] = false;
            }
        }
    }
}

vector<vector<ll>> ncr(n+1,vector<ll>(n+1,0));
for (ll i=1; i<=n; i++){
    ncr[i][0]=1;
    ncr[i][i]=1;
    for (ll j=1; j<i; j++){
        ncr[i][j]=(ncr[i-1][j]+ncr[i-1][j-1])%mod;
    }
}
