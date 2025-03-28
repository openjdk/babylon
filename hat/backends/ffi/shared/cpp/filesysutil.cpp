/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
 #include <map>
#include <mutex>
#include "filesysutil.h"
#include "strutil.h"
//#define TRACEMEM
#ifdef TRACEMEM

struct item{
   const char * tag;
   void *ptr;
   size_t size;
};
#define HEAP 100000
#define HEAPSIZE (HEAP*sizeof(item))
static item *items = new item[HEAP];
static int itemc =0;
static long heapUsage = 0;
const char *grabTag = "general";

std::recursive_mutex  heapRecursiveMutex;

void lockheap() {
   // std::cerr << "about to acquire lock " << index(std::this_thread::get_id()) << std::endl;
   while (!heapRecursiveMutex.try_lock()) {
      //   std::cerr << "Looks like we are contended!" << std::endl;
      //  std::cerr.flush();
      ::usleep(1000);
   }
   // recursiveMutex.lock();
   // std::cerr << "got lock " << index(std::this_thread::get_id()) << std::endl;

}

void unlockheap() {
   heapRecursiveMutex.unlock();
   //std::cerr << "just released lock " << index(std::this_thread::get_id()) << std::endl;
}

void *grab(const char *tag, size_t bytes){

   if (bytes == HEAPSIZE){
      return malloc(bytes);
   }

   if (itemc == 0){
      ::memset(items, 0, HEAPSIZE);
   }
   lockheap();
   items[itemc].size =bytes;
   items[itemc].tag =tag;
   heapUsage+=bytes;
 //  std::cout << "allocating slot "<<itemc<< " " << bytes<< " from "<< tag << std::endl;
   void *ptr = items[itemc++].ptr =malloc(bytes);
   unlockheap();
   return ptr;
}

void * operator new(size_t bytes){
   return grab(grabTag, bytes);
}
void operator delete(void *bytes){

   for (int i=0; i< itemc; i++){
      if (items[i].ptr == bytes){
       //  std::cout << "freeing "<<items[i].size<<" from "<<i<< " " << items[i].tag << std::endl;
         lockheap();
         heapUsage-=items[i].size;
         free(bytes);
         items[i].ptr = nullptr;
         items[i].size=0;

         unlockheap();
         std::cout << "delta "<<heapUsage<< std::endl;
         return;
      }
   }

      std::cout << "no allocation for "<< (long)bytes<<std::endl;
}
#endif
void FileSysUtil::visit(const std::string &dirName, bool recurse, std::function<void(bool dir, std::string name)> visitor) {
    DIR *d;
    if ((d = opendir(dirName.c_str())) != nullptr) {
        struct dirent *ent;
        while ((ent = readdir(d)) != nullptr) {
            std::string name = dirName + "/" + ent->d_name;
            if (ent->d_type & DT_REG) {
                visitor(false, name);
            } else if (std::strcmp(ent->d_name, ".") != 0 && std::strcmp(ent->d_name, "..") != 0 &&
                       ent->d_type & DT_DIR) {
                visitor(true, name);
                if (recurse) {
                    visit(name, recurse, visitor);
                }
            }
        }
        closedir(d);
    }
}

void FileSysUtil::forEachFileName(const std::string &dirName, std::function<void(std::string name)> visitor) {
    DIR *d;
    if ((d = opendir(dirName.c_str())) != nullptr) {
        struct dirent *ent;
        while ((ent = readdir(d)) != nullptr) {
            std::string name = dirName + "/" + ent->d_name;
            if (ent->d_type & DT_REG) {
                visitor(name);
            }
        }
        closedir(d);
    }
}

void FileSysUtil::forEachDirName(const std::string &dirName, std::function<void(std::string name)> visitor) {
    DIR *d;
    if ((d = opendir(dirName.c_str())) != nullptr) {
        struct dirent *ent;
        while ((ent = readdir(d)) != nullptr) {
            std::string name = dirName + "/" + ent->d_name;
            if (ent->d_type & DT_DIR && std::strcmp(ent->d_name, ".") != 0 && std::strcmp(ent->d_name, "..") != 0) {
                visitor(name);

            }
            closedir(d);
        }
    }
}

void FileSysUtil::forEachLine(const std::string &fileName, std::function<void(std::string name)> visitor) {
    std::size_t current, previous = 0;
    std::string content = getFile(fileName);
    current = content.find('\n');
    while (current != std::string::npos) {
        visitor(std::string(content, previous, current - previous));
        previous = current + 1;
        current = content.find('\n', previous);
    }
}

#define BUF_SIZE 4096

void FileSysUtil::send(int from, size_t bytes, int to) {
    char buf[BUF_SIZE];
    size_t bytesRead;
    size_t totalSent = 0;
    size_t bytesSent;
    while (bytes > 0
           && (((bytesRead = read(from, buf, ((bytes < BUF_SIZE) ? bytes : BUF_SIZE)))) > 0)
           && (((bytesSent = ::write(to, buf, bytesRead))) > 0)) {
        bytes -= bytesRead;
        totalSent += bytesRead;
    }
    if (bytesSent == 0) {
        perror("sendfile: send() transferred 0 bytes");
    }
}

void FileSysUtil::send(const std::string &fileName, int to) {
    int fd = ::open(fileName.c_str(), O_RDONLY);
    size_t bytes = FileSysUtil::size(fileName);
    send(fd, bytes, to);
    ::close(fd);
}


size_t FileSysUtil::size(const std::string &fileName) {

    struct stat st;
    stat(fileName.c_str(), &st);
    return st.st_size;
}

bool FileSysUtil::isDir(const std::string &dirName) {
    struct stat buffer;
    return (stat(dirName.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

bool FileSysUtil::removeFile(const std::string &dirName) {
    struct stat buffer;
    if (stat(dirName.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode)){
        std::cerr << "removing file '"+dirName<<"'"<<std::endl;
        return (::unlink(dirName.c_str()) == 0);
    }
    return false;
}

bool FileSysUtil::isFile(const std::string &fileName) {
    struct stat buffer;
    return (stat(fileName.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}
bool FileSysUtil::isFileOrLink(const std::string &fileName) {
    struct stat buffer;
    return (stat(fileName.c_str(), &buffer) == 0 && (S_ISREG(buffer.st_mode) || (S_ISLNK(buffer.st_mode))));
}

bool FileSysUtil::isFile(const std::string &dirName, const std::string &fileName) {
    std::string path = dirName + "/" + fileName;
    return isFile(path);
}
bool FileSysUtil::isFileOrLink(const std::string &dirName, const std::string &fileName) {
    std::string path = dirName + "/" + fileName;
    return isFileOrLink(path);
}

bool FileSysUtil::hasFileSuffix(const std::string &fileName, const std::string &suffix) {
    return StringUtil::endsWith(fileName, suffix);
}

std::string FileSysUtil::getFileNameEndingWith(const std::string &dir, const std::string &suffix) {
    std::vector<std::string> matches;
    visit(dir, false, [&](auto dir, auto n) { if (!dir && hasFileSuffix(n, suffix)) matches.push_back(n); });
    if (matches.size() == 0) {
        std::cout << "no file: *" << suffix << std::endl;
    } else if (matches.size() > 1) {
        std::cout << "many : *" << suffix << std::endl;
    } else {
        return *matches.begin();
    }
    return "";
}

void FileSysUtil::mkdir_p(char *path) {
    char *sep = std::strrchr(path, '/');
    if (sep != NULL) {
        *sep = 0;
        mkdir_p(path);
        *sep = '/';
    }
    if (mkdir(path, 0777) && errno != EEXIST) {
        printf("error while trying to create '%s'\n%m\n", path);
    }
}

std::string FileSysUtil::getFile(const std::string &path) {
    std::stringstream buf;
    std::ifstream input(path.c_str());
    buf << input.rdbuf();
    return buf.str();
}

BufferCursor *FileSysUtil::getFileBufferCursor(const std::string &path) {
    size_t s = size(path);
    // read directly into buffer!  buffer(path.c_str());
    char *buf = (char *)malloc(s+1);
    BufferCursor *buffer = new BufferCursor(buf, s + 1);
    int fd = open(path.c_str(), O_RDONLY);
    ::read(fd, buffer->getStart(), buffer->getSize());
    close(fd);
    return buffer;
}

void FileSysUtil::putFile(const std::string &path, const std::string &content) {
    std::ofstream out(path);
    out << content;
    out.close();
}

void FileSysUtil::putFileBufferCursor(const std::string &path, BufferCursor *buffer) {
     std::cerr << "who the hell called putFileBUffer" << std::endl;
     ::exit(1);
}
