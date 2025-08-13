package loadgen

import (
	"encoding/base64"
	"io"
	"math/rand"
	"net/http"
	"strconv"
)

type DataProvider interface {
	GetData() []byte
}

type EchoDataProvider struct {
	minSize    int
	maxSize    int
	randomData []byte
	poolSize   int
}

func NewEchoDataProvider(minSize int, maxSize int) *EchoDataProvider {
	// Pre-generate a pool of random data
	poolSize := 1024
	randomData := make([]byte, poolSize)
	for i := 0; i < poolSize; i += 8 {
		u := rand.Uint64()
		randomData[i+0] = byte(u)
		randomData[i+1] = byte(u >> 8)
		randomData[i+2] = byte(u >> 16)
		randomData[i+3] = byte(u >> 24)
		randomData[i+4] = byte(u >> 32)
		randomData[i+5] = byte(u >> 40)
		randomData[i+6] = byte(u >> 48)
		randomData[i+7] = byte(u >> 56)
	}

	return &EchoDataProvider{
		minSize:    minSize,
		maxSize:    maxSize,
		randomData: randomData,
		poolSize:   poolSize,
	}
}

func (e *EchoDataProvider) GetData() []byte {
	minSize := e.minSize
	if minSize < 8 {
		minSize = 8
	}
	maxSize := e.maxSize
	if maxSize < minSize {
		maxSize = minSize
	}
	size := rand.Intn((maxSize-minSize)+1) + minSize
	size = size &^ 7 // round down to nearest multiple of 8
	if size < 8 {
		size = 8
	}

	// Get random offset into our pre-generated data
	maxOffset := len(e.randomData) - size
	if maxOffset <= 0 {
		maxOffset = 1
	}
	offset := rand.Intn(maxOffset)

	data := make([]byte, size)
	copy(data, e.randomData[offset:offset+size])
	return data
}

/*
this is what bfs json expects

	type InputData struct {
		Size int
	}
*/
type BFSJSONDataProvider struct {
	minSize int
	maxSize int
}

func NewBFSJSONDataProvider(minSize int, maxSize int) *BFSJSONDataProvider {
	return &BFSJSONDataProvider{
		minSize: minSize,
		maxSize: maxSize,
	}
}

func (b *BFSJSONDataProvider) GetData() []byte {
	size := rand.Intn((b.maxSize-b.minSize)+1) + b.minSize
	buf := make([]byte, 0, 16) // 16 is enough for {"Size":<int>}
	buf = append(buf, '{', '"', 'S', 'i', 'z', 'e', '"', ':')
	buf = strconv.AppendInt(buf, int64(size), 10)
	buf = append(buf, '}')
	return buf
}

func fetchThumbnailerImage() ([]byte, error) {
	resp, err := http.Get("http://picsum.photos/1920/1080")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

/*
this is what thumbnailer json expects

	type InputData struct {
		Image  []byte `json:"image"`
		Width  int    `json:"width"`
		Height int    `json:"height"`
	}
*/
type ThumbnailerJSONDataProvider struct {
	image []byte
}

func NewThumbnailerJSONDataProvider() *ThumbnailerJSONDataProvider {
	img, err := fetchThumbnailerImage()
	if err != nil {
		panic(err)
	}
	return &ThumbnailerJSONDataProvider{
		image: img,
	}
}

func (t *ThumbnailerJSONDataProvider) GetData() []byte {
	width := rand.Intn(1440) + 1
	height := rand.Intn(900) + 1

	b64Len := base64.StdEncoding.EncodedLen(len(t.image))
	buf := make([]byte, 0, b64Len+64)
	buf = append(buf, '{', '"', 'i', 'm', 'a', 'g', 'e', '"', ':', '"')
	b64 := make([]byte, b64Len)
	base64.StdEncoding.Encode(b64, t.image)
	buf = append(buf, b64...)
	buf = append(buf, '"', ',')
	buf = append(buf, '"', 'w', 'i', 'd', 't', 'h', '"', ':')
	buf = strconv.AppendInt(buf, int64(width), 10)
	buf = append(buf, ',')
	buf = append(buf, '"', 'h', 'e', 'i', 'g', 'h', 't', '"', ':')
	buf = strconv.AppendInt(buf, int64(height), 10)
	buf = append(buf, '}')
	return buf
}
