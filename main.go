package main

import (
	"encoding/json"
	"fmt"
	"github.com/milvus-io/milvus-sdk-go/milvus"
	"math/rand"
	"strconv"
	"strings"
	"time"
)

var (
	HOST       = "35.228.198.36"
	PORT       = 19530
	COLLECTION = "images"

	// Test data
	dim        = 1256
	numVectors = 1500000
	// Params to tune
	fileSize         = 1024
	indexType        = milvus.IVFSQ8
	insertBatch      = 5000
	numSearchResults = 5000
	searchParams     = "{\"nprobe\" : 32}"
)

var client milvus.MilvusClient

func main() {
	Connect()
	CreateCollection()
	InsertVectors()
	CreateIndex()
	PrintCollectionInfo()
	Search()
}

func Connect() {
	var grpcClient milvus.Milvusclient
	client = milvus.NewMilvusClient(grpcClient.Instance)

	err := client.Connect(milvus.ConnectParam{IPAddress: HOST, Port: strconv.Itoa(PORT)})
	if err != nil {
		println("client: connect failed: " + err.Error())
	}
	// print created collections
	collections, status, err := client.ListCollections()
	JudgeStatus("ListCollections", status, err)
	fmt.Println("ListCollections: ", strings.Join(collections, ", "))
}

// CreateCollection creates collection if it's not exist.
func CreateCollection() {
	info, status, err := client.GetCollectionInfo(COLLECTION)
	JudgeStatus("CollectionInfo", status, err)
	if !status.Ok() {
		//create collection
		status, err = client.CreateCollection(milvus.CollectionParam{
			CollectionName: COLLECTION,
			Dimension:      int64(dim),
			IndexFileSize:  int64(fileSize),
			MetricType:     int64(milvus.IP),
		})
		JudgeStatus("CreateCollection", status, err)

		// retrieve collection info
		info, status, err = client.GetCollectionInfo(COLLECTION)
		JudgeStatus("CollectionInfo", status, err)
	}
	fmt.Println("Info: \n", string(Json(info)))
}

// InsertVectors insert N vectors in batches
func InsertVectors() {
	println("Inserting vectors...")
	for i := 0; i < numVectors/insertBatch; i++ {
		// create batch
		records := make([]milvus.Entity, insertBatch)
		for i := range records {
			vec := make([]float32, dim)
			for j := range vec {
				vec[j] = rand.Float32()
			}
			records[i].FloatData = vec
		}
		// insert
		insertParam := milvus.InsertParam{CollectionName: COLLECTION, RecordArray: records}
		id_array, status, err := client.Insert(&insertParam)
		if err != nil {
			println("InsertVectors rpc failed: " + err.Error())
			return
		}
		if !status.Ok() {
			println("InsertVectors vector failed: " + status.GetMessage())
			return
		}
		if len(id_array) != insertBatch {
			println("ERROR: return id array is null")
		}
		fmt.Printf(" [%d/%d]", i, numVectors/insertBatch)
	}
	status, err := client.Flush([]string{COLLECTION})
	JudgeStatus("Flush", status, err)

	println("InsertVectors vectors success!")
}

// Search finds topk results for a random vector and prints the requests time.
func Search() {
	numSearches := 10
	queryRecords := make([]milvus.Entity, numSearches)
	for i := range queryRecords {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		queryRecords[i].FloatData = vec
	}
	for _, r := range queryRecords {
		start := time.Now()
		_, status, err := client.Search(milvus.SearchParam{
			CollectionName: COLLECTION,
			QueryEntities:  []milvus.Entity{r},
			Topk:           int64(numSearchResults),
			ExtraParams:    searchParams,
		})
		fmt.Println(time.Now().Sub(start))
		JudgeStatus("GetCollectionStats", status, err)
	}
}

// CreateIndex creates an index if it needs to be changed.
func CreateIndex() {
	indexParam, status, err := client.GetIndexInfo(COLLECTION)
	JudgeStatus("GetIndexInfo", status, err)
	fmt.Println("Current index:", Json(indexParam))

	if indexParam.IndexType == indexType {
		return
	}

	fmt.Println("Create index started")
	start := time.Now()
	status, err = client.CreateIndex(&milvus.IndexParam{
		CollectionName: COLLECTION,
		IndexType:      indexType,
		ExtraParams:    "{\"nlist\" : 16384}",
	})
	JudgeStatus("", status, err)
	fmt.Println("Create index finished:", time.Now().Sub(start))
}

func PrintCollectionInfo() {
	stats, status, err := client.GetCollectionStats(COLLECTION)
	JudgeStatus("GetCollectionStats", status, err)
	fmt.Println("Stats: ", stats)
}

func Json(v interface{}) []byte {
	data, _ := json.Marshal(v)
	return data
}

func JudgeStatus(funcName string, status milvus.Status, err error) {
	if err != nil {
		println(funcName + " rpc failed: " + err.Error())
		return
	}
	if !status.Ok() {
		println(funcName + " failed: " + status.GetMessage())
		return
	}
}
