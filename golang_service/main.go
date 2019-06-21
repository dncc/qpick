package main

/*
#cgo LDFLAGS: -L./lib -lgoqpick
#include "./lib/goqpick.h"
*/
import "C"

import (
	"flag"
	"github.com/gorilla/mux"
	"github.com/gorilla/rpc"
	"github.com/gorilla/rpc/json"
	"log"
	"net/http"
)

const (
	PORT_DEFAULT = "8888"
)

var (
	port      *string
	indexPath *string
	qpick     *C.struct_Qpick
)

//Holds arguments to be passed to service QPickRPCService in RPC call
type Args struct {
	Q string
	C uint32
	TFIDF uint8
}

//Represents service QPickRPCService with method Multiply
type QPickRPCService int

//Result of RPC call is of this type
// type Result int
type Result string

func (t *QPickRPCService) Get(r *http.Request, args *Args, result *Result) error {
	var res = C.qpick_get_as_string(qpick, C.CString(args.Q), C.uint32_t(args.C), C.uint8_t(args.TFIDF))
	*result = Result(C.GoString(res))
	return nil
}

func init() {
	port = flag.String("port", PORT_DEFAULT, "Main port for serving requests; defaults to 8888 if unspecified")
	indexPath = flag.String("index", "", "REQUIRED. Directory path of the qpick index")
}

func main() {
	flag.Parse()
	log.Printf("Running on: %v", *port)
	log.Printf("Index path: %v", *indexPath)
	qpick = C.qpick_init(C.CString(*indexPath))
	s := rpc.NewServer()
	s.RegisterCodec(json.NewCodec(), "application/json")
	s.RegisterCodec(json.NewCodec(), "application/json;charset=UTF-8")
	qpick := new(QPickRPCService)
	s.RegisterService(qpick, "")
	r := mux.NewRouter()
	r.Handle("/rpc", s)
	http.ListenAndServe(":"+*port, r)
}
